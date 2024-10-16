
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import logging
import warnings
import pickle
import random
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import traceback
import torch.optim.lr_scheduler as lr_scheduler


# In[2]:


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class EHRDataPreprocessor:
    def __init__(self, data_path: str, feature_files: List[str], demographic_file: str, max_seq_length: int):
        self.data_path = Path(data_path)
        self.feature_files = feature_files
        self.demographic_file = demographic_file
        self.max_seq_length = max_seq_length
        self.data_types = self._get_data_types()
        self.data_type_to_idx = {dt: idx for idx, dt in enumerate(self.data_types)}
        self.value_normalizer = ValueNormalizer()
        self.reference_date = datetime(2000, 1, 1)
        self.continuous_features = ['hba1c', 'sbp', 'bmi', 'creat']
        self.categorical_features = []

    def _get_data_types(self) -> List[str]:
        return [file.split('.')[0] for file in self.feature_files] + ['demographics']

    def load_and_preprocess_data(self) -> Dict[int, Dict[str, torch.Tensor]]:
        feature_data = self._load_feature_data()
        demographic_data = self._load_demographic_data()
        
        logging.info("Fitting normalizers...")
        self.value_normalizer.fit(feature_data, demographic_data)
        
        logging.info("Preprocessing data...")
        preprocessed_data = {}
        for patient_id in tqdm(feature_data.keys(), desc="Processing patients"):
            patient_feature_data = feature_data[patient_id]
            patient_demographic_data = demographic_data.get(patient_id, {})
            try:
                preprocessed_data[patient_id] = self.preprocess_patient_data(patient_id, patient_feature_data, patient_demographic_data)
            except Exception as e:
                logging.error(f"Error processing patient {patient_id}: {str(e)}")
        
        return preprocessed_data

    def _load_feature_data(self) -> Dict[int, List[Tuple[str, float, float]]]:
        feature_data = {}
        for file in self.feature_files:
            df = pd.read_csv(self.data_path / file)
            feature_name = file.split('.')[0]
            for _, row in df.iterrows():
                patient_id = int(row['patient_id'])
                if patient_id not in feature_data:
                    feature_data[patient_id] = []
                timestamp = (datetime.strptime(row['timestamp'], '%Y-%m-%d') - self.reference_date).days
                value = row['value'] if pd.notna(row['value']) else None
                feature_data[patient_id].append((feature_name, value, float(timestamp)))
        return feature_data

    def _load_demographic_data(self) -> Dict[int, Dict[str, str]]:
        df = pd.read_csv(self.data_path / self.demographic_file)
        return {int(row['patient_id']): {col: str(row[col]) for col in df.columns if col != 'patient_id'} for _, row in df.iterrows()}

    def preprocess_patient_data(self, patient_id: int, feature_data: List[Tuple[str, float, float]], demographic_data: Dict[str, str]) -> Dict[str, torch.Tensor]:
        feature_data.sort(key=lambda x: x[2])
        seq_length = min(len(feature_data) + 1, self.max_seq_length)
        data_type_tensor = torch.zeros(seq_length, dtype=torch.long)
        value_tensor = torch.zeros(seq_length)
        time_tensor = torch.zeros(seq_length)
        mask_tensor = torch.zeros(seq_length, dtype=torch.bool)
        missing_tensor = torch.zeros(seq_length, dtype=torch.bool)

        for i, (data_type, value, timestamp) in enumerate(feature_data[:seq_length - 1]):
            data_type_tensor[i] = self.data_type_to_idx[data_type]
            if value is not None:
                value_tensor[i] = self.value_normalizer.normalize(data_type, value)
                mask_tensor[i] = True
            else:
                missing_tensor[i] = True
            time_tensor[i] = timestamp

        demo_idx = seq_length - 1
        data_type_tensor[demo_idx] = self.data_type_to_idx['demographics']
        value_tensor[demo_idx] = self.value_normalizer.normalize_demographics(demographic_data)
        time_tensor[demo_idx] = time_tensor[demo_idx - 1] if demo_idx > 0 else 0
        mask_tensor[demo_idx] = True

        time_diff_tensor = torch.zeros_like(time_tensor)
        time_diff_tensor[1:] = time_tensor[1:] - time_tensor[:-1]

        # Move tensors to GPU
        return {
            'patient_id': patient_id,
            'data_type': data_type_tensor,
            'value': value_tensor,
            'timestamp': time_tensor,
            'time_diff': time_diff_tensor,
            'mask': mask_tensor,
            'missing': missing_tensor
        }

class ValueNormalizer:
    def __init__(self):
        self.feature_scalers = {}
        self.demographic_encoder = None
        self.demographic_scaler = None
        self.reference_date = datetime(2000, 1, 1)
        self.demographic_columns = []
        self.data_type_to_idx = {}
        self.idx_to_data_type = {}

    def fit(self, feature_data: Dict[int, List[Tuple[str, float, float]]], demographic_data: Dict[int, Dict[str, str]]):
        feature_values = {feature: [] for feature in set(data_type for patient in feature_data.values() for data_type, _, _ in patient)}
        for patient_data in feature_data.values():
            for data_type, value, _ in patient_data:
                if value is not None:
                    feature_values[data_type].append(value)
        
        for feature, values in feature_values.items():
            self.feature_scalers[feature] = StandardScaler().fit(np.array(values).reshape(-1, 1))

        demographic_df = pd.DataFrame(demographic_data).T
        self.demographic_columns = demographic_df.columns.tolist()

        # Check if 'DOB' and 'date_diagnosis' columns are present
        if 'DOB' in demographic_df.columns:
            demographic_df['age'] = (pd.to_datetime(demographic_df['DOB'], errors='coerce') - self.reference_date).dt.days / 365.25
        else:
            demographic_df['age'] = 0  # Default value if 'DOB' is not present

        if 'date_diagnosis' in demographic_df.columns:
            demographic_df['years_since_diagnosis'] = (pd.to_datetime(demographic_df['date_diagnosis'], errors='coerce') - self.reference_date).dt.days / 365.25
        else:
            demographic_df['years_since_diagnosis'] = 0  # Default value if 'date_diagnosis' is not present

        categorical_columns = [col for col in ['dm_type', 'sex', 'ethCode'] if col in demographic_df.columns]
        
        if categorical_columns:
            self.demographic_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.demographic_encoder.fit(demographic_df[categorical_columns])
        
        numerical_columns = ['age', 'years_since_diagnosis']
        numerical_columns = [col for col in numerical_columns if col in demographic_df.columns]
        
        if numerical_columns:
            self.demographic_scaler = StandardScaler().fit(demographic_df[numerical_columns].fillna(0))
        
        # Create bidirectional mappings
        self.data_type_to_idx = {data_type: idx for idx, data_type in enumerate(self.feature_scalers.keys())}
        self.idx_to_data_type = {idx: data_type for data_type, idx in self.data_type_to_idx.items()}

    def normalize(self, data_type: str, value: float) -> float:
        if data_type in self.feature_scalers:
            return self.feature_scalers[data_type].transform([[value]])[0][0]
        else:
            return value
        
    def inverse_normalize(self, data_type: str, value: float) -> float:
        if data_type in self.feature_scalers:
            return self.feature_scalers[data_type].inverse_transform([[value]])[0][0]
        else:
            return value

    def normalize_demographics(self, demographic_data: Dict[str, str]) -> float:
        try:
            age = 0
            years_since_diagnosis = 0
            
            if 'DOB' in demographic_data:
                dob = pd.to_datetime(demographic_data['DOB'], errors='coerce')
                age = (self.reference_date - dob).days / 365.25 if pd.notnull(dob) else 0
            
            if 'date_diagnosis' in demographic_data:
                diagnosis_date = pd.to_datetime(demographic_data['date_diagnosis'], errors='coerce')
                years_since_diagnosis = (self.reference_date - diagnosis_date).days / 365.25 if pd.notnull(diagnosis_date) else 0
            
            categorical_data = []
            for col in ['dm_type', 'sex', 'ethCode']:
                if col in demographic_data:
                    categorical_data.append(demographic_data[col])
                else:
                    categorical_data.append('Unknown')
            
            all_features = []
            if self.demographic_encoder is not None:
                encoded_categorical = self.demographic_encoder.transform([categorical_data])
                all_features.extend(encoded_categorical.flatten())
            
            if self.demographic_scaler is not None:
                scaled_numerical = self.demographic_scaler.transform([[age, years_since_diagnosis]])
                all_features.extend(scaled_numerical.flatten())
            
            if not all_features:
                return 0.0  # Return a default value if no features are available
            
            return float(np.mean(all_features))
        except Exception as e:
            logging.error(f"Error in normalize_demographics: {str(e)}")
            logging.error(f"Demographic data: {demographic_data}")
            return 0.0
            
        def inverse_normalize(self, data_type: str, value: float) -> float:
            if data_type in self.feature_scalers:
                return self.feature_scalers[data_type].inverse_transform([[value]])[0][0]
            else:
                return value

class EHRDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data
        self.patient_ids = list(preprocessed_data.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_data = self.data[patient_id]
        
        # Ensure demographic data is included
        demographic_data = {
            'DOB': patient_data.get('DOB', None),
            'dm_type': patient_data.get('dm_type', None),
            'date_diagnosis': patient_data.get('date_diagnosis', None),
            'sex': patient_data.get('sex', None),
            'ethCode': patient_data.get('ethCode', None)
        }
        
        # Create attention_mask if not present
        if 'attention_mask' not in patient_data:
            attention_mask = patient_data['mask'].clone()
        else:
            attention_mask = patient_data['attention_mask']
        
        # Calculate lengths if not present
        if 'lengths' not in patient_data:
            lengths = patient_data['mask'].sum().item()
        else:
            lengths = patient_data['lengths']
        
        return {
            'patient_id': patient_id,
            'data_type': patient_data['data_type'],
            'value': patient_data['value'],
            'timestamp': patient_data['timestamp'],
            'time_diff': patient_data['time_diff'],
            'mask': patient_data['mask'],
            'missing': patient_data['missing'],
            'attention_mask': attention_mask,
            'lengths': lengths,
            **demographic_data  # Include demographic data
        }
     
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['data_type']), reverse=True)
    
    lengths = [len(item['data_type']) for item in batch]
    max_len = max(lengths)

    padded_batch = {
        'patient_id': torch.tensor([item['patient_id'] for item in batch]),
        'data_type': pad_sequence([item['data_type'] for item in batch], batch_first=True, padding_value=0),
        'value': pad_sequence([item['value'] for item in batch], batch_first=True, padding_value=0.0),
        'timestamp': pad_sequence([item['timestamp'] for item in batch], batch_first=True, padding_value=0.0),
        'time_diff': pad_sequence([item['time_diff'] for item in batch], batch_first=True, padding_value=0.0),
        'mask': pad_sequence([item['mask'] for item in batch], batch_first=True, padding_value=False),
        'missing': pad_sequence([item['missing'] for item in batch], batch_first=True, padding_value=True)
    }

    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    padded_batch['attention_mask'] = attention_mask
    padded_batch['lengths'] = torch.tensor(lengths)

    return padded_batch  # Return CPU tensors

class PatientSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.patient_to_indices = self._create_patient_to_indices()

    def _create_patient_to_indices(self):
        patient_to_indices = {}
        for idx, item in enumerate(self.dataset.data):
            patient_id = item['patient_id']
            if patient_id not in patient_to_indices:
                patient_to_indices[patient_id] = []
            patient_to_indices[patient_id].append(idx)
        return patient_to_indices

    def __iter__(self):
        batches = []
        patients = list(self.patient_to_indices.keys())
        random.shuffle(patients)
        
        current_batch = []
        for patient in patients:
            indices = self.patient_to_indices[patient]
            if len(indices) >= 2:
                # Add two instances of the same patient to ensure positive pairs
                current_batch.extend(random.sample(indices, 2))
            elif len(indices) == 1:
                # If only one instance, add it twice
                current_batch.extend(indices * 2)
            
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch[:self.batch_size])
                current_batch = current_batch[self.batch_size:]
        
        if not self.drop_last and current_batch:
            batches.append(current_batch)
        
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
               
class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, time_diff, mask=None):
        batch_size = query.size(0)
        seq_length = query.size(1)
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        time_diff = time_diff.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_length]
        time_diff = time_diff.expand(-1, self.num_heads, seq_length, -1)  # [batch_size, num_heads, seq_length, seq_length]
        time_diff = torch.abs(time_diff.transpose(-1, -2) - time_diff)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        time_impact = torch.exp(-time_diff.clamp(min=-100, max=100))
        scores = scores * time_impact
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, 0.0, 0.0, 0.0)  # Replace NaNs with zeros
        context = torch.matmul(attn, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.W_o(context)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, timestamps):
        x = x + self.pe[:x.size(0), :]
        # print(f"In PositionalEncoding, x shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = TimeAwareMultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, time_diff, mask=None):
        # Apply layer norm before self-attention
        x_norm = self.norm1(x)
        
        attn_output = self.self_attn(x_norm, x_norm, x_norm, time_diff, mask)
        # print(f"After self-attention, output shape: {attn_output.shape}, min: {attn_output.min().item()}, max: {attn_output.max().item()}")
        
        # Add residual connection
        x = x + self.dropout(attn_output)
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)  # Replace NaNs with zeros
        # print(f"After residual connection, x shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")
        
        # Apply layer norm before feed-forward
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        # print(f"After feed-forward, output shape: {ff_output.shape}, min: {ff_output.min().item()}, max: {ff_output.max().item()}")
        
        # Add residual connection
        x = x + self.dropout(ff_output)
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)  # Replace NaNs with zeros
        # print(f"After second residual connection, x shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")
        
        return x

class EHRTransformer(nn.Module):
    def __init__(self, num_data_types, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.data_type_embedding = nn.Embedding(num_data_types, d_model)
        self.value_embedding = nn.Linear(1, d_model)
        self.time_embedding = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Use Pre-LN Transformer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, data_type, value, timestamp, time_diff, mask):
        # Embeddings
        data_type_emb = self.data_type_embedding(data_type)
        value_emb = self.value_embedding(value.unsqueeze(-1))
        time_emb = self.time_embedding(timestamp.unsqueeze(-1))
        
        # Combine embeddings
        x = data_type_emb + value_emb + time_emb
        x = self.layer_norm(x)
        
        # Apply transformer
        mask = ~mask  # Invert mask for transformer
        output = self.transformer(x, src_key_padding_mask=mask)
        
        # Final output
        reconstructed_values = self.fc_out(output).squeeze(-1)
        
        return {
            'reconstructed_values': reconstructed_values,
            'embeddings': output.mean(dim=1)  # Use mean pooling for embeddings
        }
    
class MaskedReconstructionLoss(nn.Module):
    def __init__(self, continuous_features, num_categories):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.continuous_features = continuous_features
        self.num_categories = num_categories

    def forward(self, predictions, targets, data_type, mask_indices, attention_mask):
        active_loss = (attention_mask.view(-1) == 1) & (mask_indices.view(-1) == 1)
        active_preds = predictions.view(-1, predictions.size(-1))[active_loss]
        active_targets = targets.view(-1)[active_loss]
        active_data_type = data_type.view(-1)[active_loss]

        cont_mask = torch.tensor([dt in self.continuous_features for dt in active_data_type])
        cat_mask = ~cont_mask

        total_loss = 0
        num_samples = active_loss.sum()

        if cont_mask.any():
            cont_preds = active_preds[cont_mask, 0]
            cont_targets = active_targets[cont_mask]
            cont_loss = self.mse_loss(cont_preds, cont_targets).mean()
            total_loss += cont_loss
            # print(f"Continuous Loss: {cont_loss.item():.4f}")
            # print(f"Continuous Preds min/max: {cont_preds.min().item():.4f}/{cont_preds.max().item():.4f}")
            # print(f"Continuous Targets min/max: {cont_targets.min().item():.4f}/{cont_targets.max().item():.4f}")

        if cat_mask.any():
            cat_preds = active_preds[cat_mask]
            cat_targets = active_targets[cat_mask].long()
            cat_targets = torch.clamp(cat_targets, min=0, max=self.num_categories-1)
            cat_loss = self.ce_loss(cat_preds, cat_targets).mean()
            total_loss += cat_loss
            # print(f"Categorical Loss: {cat_loss.item():.4f}")
            # print(f"Categorical Preds min/max: {cat_preds.min().item():.4f}/{cat_preds.max().item():.4f}")
            # print(f"Categorical Targets min/max: {cat_targets.min().item()}/{cat_targets.max().item()}")

        return total_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, patient_ids):
        if torch.isnan(embeddings).any():
            print("NaN detected in embeddings!")
            return torch.tensor(0.0, requires_grad=True)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        pos_mask = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
        
        if pos_mask.sum() == 0:
            print("No positive pairs found in the batch!")
            return torch.tensor(0.0, requires_grad=True)

        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)

        pos_sim = torch.exp(sim_matrix) * pos_mask
        neg_sim = torch.exp(sim_matrix) * neg_mask

        loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
        return loss.mean()

def mask_data(data, mask, mask_ratio=0.15):
    mask_indices = torch.rand_like(data.float()) < mask_ratio
    mask_indices = mask_indices & mask
    masked_data = data.clone()
    masked_data[mask_indices] = 0
    return masked_data, mask_indices

def train_epoch(model, dataloader, optimizer, continuous_features, num_categories, device, mask_ratio=0.15, clip_value=1.0):
    model.train()
    total_loss = 0
    reconstruction_loss_fn = MaskedReconstructionLoss(continuous_features, num_categories).to(device)
    contrastive_loss_fn = ContrastiveLoss().to(device)
    
    epoch_losses = {
        'total': [],
        'reconstruction': [],
        'contrastive': []
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()

        # Move individual tensors to device
        data_type = batch['data_type'].to(device)
        value = batch['value'].to(device)
        timestamp = batch['timestamp'].to(device)
        time_diff = batch['time_diff'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        patient_ids = batch['patient_id'].to(device)

        masked_value, mask_indices = mask_data(value, attention_mask, mask_ratio)
        mask_indices = mask_indices.to(device)

        try:
            output = model(data_type, masked_value, timestamp, time_diff, attention_mask)
            
            if torch.isnan(output).any():
                print(f"NaN detected in model output at batch {batch_idx}")
                print(f"Input shapes: data_type {data_type.shape}, masked_value {masked_value.shape}, timestamp {timestamp.shape}, time_diff {time_diff.shape}, attention_mask {attention_mask.shape}")
                print(f"Output shape: {output.shape}")
                print(f"Output min/max: {output.min().item()}/{output.max().item()}")
                continue

            recon_loss = reconstruction_loss_fn(output, value, data_type, mask_indices, attention_mask)
            
            mean_embedding = (output * attention_mask.unsqueeze(-1)).sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-8)
            contrastive_loss = contrastive_loss_fn(mean_embedding, patient_ids)

            contrastive_weight = 0.1
            loss = recon_loss + contrastive_weight * contrastive_loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

                total_loss += loss.item()
                epoch_losses['total'].append(loss.item())
                epoch_losses['reconstruction'].append(recon_loss.item())
                epoch_losses['contrastive'].append(contrastive_loss.item())

            # if batch_idx % 10 == 0:
                # print(f"Batch {batch_idx} losses - Total: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, Contrastive: {contrastive_loss.item():.4f}")
                #print(f"Unique patient IDs in batch: {patient_ids.unique().shape[0]}")
                #print(f"Patient ID counts: {torch.bincount(patient_ids)}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            print("Batch keys:", batch.keys())
            for key, value in batch.items():
                print(f"{key} shape: {value.shape}")
            continue

    return total_loss / len(dataloader), epoch_losses

def train_model(model, dataloader, num_epochs, learning_rate, device, continuous_features, num_categories):
    model = model.to(device)
    print(f"Using device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    value_normalizer = ValueNormalizer()
    
    all_losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': [], 'reconstruction': [], 'contrastive': []}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move entire batch to GPU
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            output = model(batch['data_type'], batch['value'], batch['timestamp'], batch['time_diff'], batch['mask'])
            
            losses = compute_loss(output, batch)
            total_loss = losses['total']
            
            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['reconstruction'].append(losses['reconstruction'].item())
                epoch_losses['contrastive'].append(losses['contrastive'].item())
            else:
                print(f"Skipping batch due to NaN or Inf loss: {losses}")
        
        if epoch_losses['total']:
            avg_total_loss = sum(epoch_losses['total']) / len(epoch_losses['total'])
            avg_reconstruction_loss = sum(epoch_losses['reconstruction']) / len(epoch_losses['reconstruction'])
            avg_contrastive_loss = sum(epoch_losses['contrastive']) / len(epoch_losses['contrastive'])
            
            all_losses.append({
                'epoch': epoch + 1,
                'avg_loss': avg_total_loss,
                'reconstruction': avg_reconstruction_loss,
                'contrastive': avg_contrastive_loss
            })
            
            scheduler.step(avg_total_loss)
        else:
            print(f"Epoch {epoch+1}/{num_epochs} had no valid batches.")

    return model, all_losses, value_normalizer

def plot_losses(losses):
    epochs = [loss['epoch'] for loss in losses]
    avg_losses = [loss['avg_loss'] for loss in losses]
    reconstruction_losses = [loss['reconstruction'] for loss in losses]
    contrastive_losses = [loss['contrastive'] for loss in losses]

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, avg_losses, label='Average Loss')
    plt.plot(epochs, reconstruction_losses, label='Reconstruction Loss')
    plt.plot(epochs, contrastive_losses, label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()

def reconstruct_and_compare_features(model, dataloader, device, value_normalizer, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            output = model(batch['data_type'], batch['value'], batch['timestamp'], batch['time_diff'], batch['mask'])
            
            reconstructed_values = output['reconstructed_values']
            
            print(f"Batch shapes:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.shape}")
            print(f"Reconstructed values shape: {reconstructed_values.shape}")
            
            for i in range(min(num_samples, batch['value'].shape[0])):
                patient_data = {k: v[i] for k, v in batch.items()}
                patient_reconstructed = reconstructed_values[i] if reconstructed_values.dim() > 1 else reconstructed_values
                
                valid_indices = patient_data['attention_mask'].bool()
                
                timestamps = patient_data['timestamp'][valid_indices].cpu().numpy()
                original_values = patient_data['value'][valid_indices].cpu().numpy()
                reconstructed_values = patient_reconstructed[valid_indices].cpu().numpy()
                
                ax = axes[i]
                ax.scatter(timestamps, original_values, label='Original', alpha=0.7)
                ax.scatter(timestamps, reconstructed_values, label='Reconstructed', alpha=0.7)
                ax.set_title(f'Patient {i+1}')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Value')
                ax.legend()
                
                # Annotate data types
                for j, (t, v, dt) in enumerate(zip(timestamps, original_values, patient_data['data_type'][valid_indices].cpu().numpy())):
                    data_type = value_normalizer.idx_to_data_type.get(int(dt), 'Unknown')
                    ax.annotate(data_type, (t, v), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
            
            break  # Only process one batch
    
    plt.tight_layout()
    plt.show()

def visualize_preprocessed_data(preprocessed_data: Dict[int, Dict[str, torch.Tensor]], 
                                value_normalizer: ValueNormalizer, 
                                num_patients: int = 5, 
                                max_time_points: int = 100):
    """
    Visualize preprocessed data for a few patients to check irregularity and varying lengths.
    
    :param preprocessed_data: Dictionary of preprocessed patient data
    :param value_normalizer: ValueNormalizer object used for scaling
    :param num_patients: Number of patients to visualize
    :param max_time_points: Maximum number of time points to show (to avoid overcrowding)
    """
    patient_ids = list(preprocessed_data.keys())[:num_patients]
    
    fig, axs = plt.subplots(num_patients, 1, figsize=(15, 5 * num_patients))
    fig.suptitle("Preprocessed Patient Data")
    
    for i, patient_id in enumerate(patient_ids):
        patient_data = preprocessed_data[patient_id]
        
        # Get valid indices (where mask is True)
        valid_indices = patient_data['mask'].bool()
        
        # Extract data
        timestamps = patient_data['timestamp'][valid_indices].numpy()
        data_types = patient_data['data_type'][valid_indices].numpy()
        values = patient_data['value'][valid_indices].numpy()
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        data_types = data_types[sort_idx]
        values = values[sort_idx]
        
        # Limit to max_time_points
        if len(timestamps) > max_time_points:
            timestamps = timestamps[:max_time_points]
            data_types = data_types[:max_time_points]
            values = values[:max_time_points]
        
        # Unscale values
        unscaled_values = []
        for j, data_type_idx in enumerate(data_types):
            data_type_name = value_normalizer.idx_to_data_type[data_type_idx]
            unscaled_values.append(value_normalizer.inverse_normalize(data_type_name, values[j]))
        
        # Plot
        scatter = axs[i].scatter(timestamps, unscaled_values, c=data_types, cmap='viridis')
        axs[i].set_title(f"Patient {patient_id}")
        axs[i].set_xlabel("Timestamp")
        axs[i].set_ylabel("Value")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axs[i])
        cbar.set_label('Data Type')
        
        # Annotate some points with data type names
        for j in range(0, len(timestamps), max(1, len(timestamps) // 10)):  # Annotate about 10 points
            data_type_name = value_normalizer.idx_to_data_type[data_types[j]]
            axs[i].annotate(data_type_name, (timestamps[j], unscaled_values[j]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
def downsample_data(data, fraction=0.1):
    """
    Downsample the data to a fraction of its original size.
    
    :param data: Dictionary of preprocessed patient data
    :param fraction: Fraction of data to keep (default: 0.1 for 10%)
    :return: Downsampled data
    """
    patient_ids = list(data.keys())
    num_patients_to_keep = int(len(patient_ids) * fraction)
    
    # Randomly select patients to keep
    patients_to_keep = random.sample(patient_ids, num_patients_to_keep)
    
    # Create a new dictionary with only the selected patients
    downsampled_data = {patient_id: data[patient_id] for patient_id in patients_to_keep}
    
    return downsampled_data
def compute_loss(output, batch, alpha=0.5, temperature=0.1):
    device = output['reconstructed_values'].device
    
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(output['reconstructed_values'], batch['value'], reduction='none')
    reconstruction_loss = (reconstruction_loss * batch['mask']).sum() / batch['mask'].sum()
    
    # Compute contrastive loss
    embeddings = output['embeddings']
    batch_size = embeddings.size(0)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
    
    # Create labels for contrastive loss
    labels = torch.arange(batch_size, device=device)
    
    # Compute contrastive loss
    contrastive_loss = F.cross_entropy(sim_matrix, labels)
    
    if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
        print(f"Contrastive loss is NaN or Inf. Sim matrix stats: min={sim_matrix.min().item()}, max={sim_matrix.max().item()}, mean={sim_matrix.mean().item()}")
        contrastive_loss = torch.tensor(0.0, device=device)

    total_loss = (1 - alpha) * reconstruction_loss + alpha * contrastive_loss

    return {
        'total': total_loss,
        'reconstruction': reconstruction_loss,
        'contrastive': contrastive_loss
    }

    
    


# In[3]:


# Main execution code
if __name__ == "__main__":
    # Data preprocessing
    data_path = "./processed_data"
    feature_files = ["hba1c.csv", "sbp.csv", "bmi.csv", "creat.csv"]
    demographic_file = "demographics.csv"
    max_seq_length = 40

    preprocessed_data_file = 'preprocessed_data.pkl'

    if Path(preprocessed_data_file).exists():
        logging.info(f"Loading preprocessed data from {preprocessed_data_file}")
        with open(preprocessed_data_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
    else:
        logging.info("Preprocessing data...")
        preprocessor = EHRDataPreprocessor(data_path, feature_files, demographic_file, max_seq_length)
        preprocessed_data = preprocessor.load_and_preprocess_data()
        
        logging.info(f"Saving preprocessed data to {preprocessed_data_file}")
        with open(preprocessed_data_file, 'wb') as f:
            pickle.dump(preprocessed_data, f)

    # Downsample data for debugging
    #preprocessed_data = downsample_data(preprocessed_data, fraction=0.1)
    print(f"Downsampled data size: {len(preprocessed_data)} patients")

    # Create dataset and dataloader
    dataset = EHRDataset(preprocessed_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # Model parameters
    num_data_types = 100  # Adjust based on your data
    d_model = 256
    num_heads = 16
    num_layers = 12
    d_ff = 512
    dropout = 0.4

    # Create the model
    model = EHRTransformer(num_data_types, d_model, num_heads, num_layers, d_ff, dropout).to(device)

    # Training parameters
    num_epochs = 200
    learning_rate = 1e-4  # Reduced from 1e-4
    continuous_features = ['hba1c', 'sbp', 'bmi', 'creat']
    num_categories = 40

    # Train the model
    #model, losses, value_normalizer = train_model(model, dataloader, num_epochs, learning_rate, device, continuous_features, num_categories)

    try:
        # Train the model
        model, losses, value_normalizer = train_model(model, dataloader, num_epochs, learning_rate, device, continuous_features, num_categories)

        # Plot losses
        plot_losses(losses)

        # Visualize reconstructions
        #reconstruct_and_compare_features(model, dataloader, device, value_normalizer)

        # Visualize preprocessed data
        #visualize_preprocessed_data(preprocessed_data, value_normalizer, output_dir='plots')

        print("Training and visualization complete.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

