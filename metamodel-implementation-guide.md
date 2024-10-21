# Metamodel Prediction and Recalibration System: Implementation Guide

## Overview

This document provides instructions for implementing and using the Metamodel Prediction and Recalibration System. The system processes base model outputs through a series of metamodels and applies recalibration to produce final predictions.

## Prerequisites

- Python 3.6+
- NumPy library
- Access to the following files:
  - `metamodel_coefficients.json`: Contains coefficients for all metamodels
  - `exportable_recalibrators.pkl`: Contains recalibration information for all models

## File Structure

Ensure your project directory contains the following files:

```
project_directory/
│
├── metamodel_system.py
├── metamodel_coefficients.json
└── exportable_recalibrators.pkl
```

## Implementation Steps

1. Create a new Python file named `metamodel_system.py` and copy the following code into it:

```python
import numpy as np
import json
import pickle

def load_coefficients(file_path='metamodel_coefficients.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_recalibrators(file_path='exportable_recalibrators.pkl'):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def predict_metamodel(model_name, base_predictions, coeffs):
    model_coeffs = coeffs[model_name]
    prediction = model_coeffs['intercept']
    for pred_name, value in base_predictions.items():
        prediction += model_coeffs[pred_name] * value
    return 1 / (1 + np.exp(-prediction))

def predict_and_recalibrate(original_pred, recalibration_info):
    original_pred = np.array(original_pred)
    recalibrated_pred = np.zeros_like(original_pred)
    
    for lower, upper, points in recalibration_info:
        mask = (original_pred >= lower) & (original_pred < upper)
        if points is not None and np.any(mask):
            x_values, y_values = zip(*points)
            recalibrated_pred[mask] = np.interp(original_pred[mask], x_values, y_values)
        else:
            recalibrated_pred[mask] = original_pred[mask]
    
    return recalibrated_pred

def process_all_models(base_model_outputs, coeffs, recalibrators):
    model_names = ['AMP', 'BL', 'MIM', 'MIF', 'RR', 'UKR', 'micro', 'macro', 'ANY']
    recalibrated_predictions = {}

    base_preds = {f"{name}_pred": value for name, value in zip(['AMP', 'BL', 'MIM', 'MIF', 'RR', 'UKR'], base_model_outputs)}

    recalibrators_upper = {k.upper(): v for k, v in recalibrators.items()}

    for model_name in model_names:
        original_pred = predict_metamodel(model_name, base_preds, coeffs)
        
        recalibrator_key = model_name.upper()
        if recalibrator_key in recalibrators_upper:
            recalibrated_pred = predict_and_recalibrate([original_pred], recalibrators_upper[recalibrator_key])
            recalibrated_predictions[model_name] = recalibrated_pred[0]
        else:
            print(f"Warning: No recalibrator found for {model_name}. Using original prediction.")
            recalibrated_predictions[model_name] = original_pred
    
    return recalibrated_predictions

# Main execution example
if __name__ == "__main__":
    coeffs = load_coefficients()
    recalibrators = load_recalibrators()

    test_array = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    results = process_all_models(test_array, coeffs, recalibrators)

    print("\nInput array:", test_array)
    print("\nResults:")
    for model, prediction in results.items():
        print(f"{model}: {prediction:.6f}")
```

2. Ensure that `metamodel_coefficients.json` and `exportable_recalibrators.pkl` are in the same directory as `metamodel_system.py`.

## Usage Instructions

To use the Metamodel Prediction and Recalibration System in your existing pipeline:

1. Import the necessary functions in your main pipeline script:

```python
from metamodel_system import load_coefficients, load_recalibrators, process_all_models
```

2. Load the coefficients and recalibrators at the beginning of your script or in your initialization function:

```python
coeffs = load_coefficients()
recalibrators = load_recalibrators()
```

3. When you have the base model outputs for an individual, use the `process_all_models` function to get the recalibrated predictions:

```python
base_model_outputs = [amp_pred, bl_pred, mim_pred, mif_pred, rr_pred, ukr_pred]
results = process_all_models(base_model_outputs, coeffs, recalibrators)
```

4. The `results` dictionary will contain the recalibrated predictions for all models, including the composite models (micro, macro, ANY).

## Important Notes

- The system expects 6 base model outputs in the order: AMP, BL, MIM, MIF, RR, UKR.
- The system handles case sensitivity for model names, so 'micro', 'macro', and 'ANY' will correctly match their uppercase counterparts in the recalibrators.
- If a recalibrator is not found for a model, the system will use the original prediction and print a warning.
- Ensure that the structure of `metamodel_coefficients.json` and `exportable_recalibrators.pkl` matches the expected format used in the functions.

## Troubleshooting

- If you encounter file not found errors, check that the paths to `metamodel_coefficients.json` and `exportable_recalibrators.pkl` are correct.
- If you get key errors, ensure that all expected models are present in both the coefficients and recalibrators files.
- For any other issues, check the console output for warnings or error messages.

## Maintenance and Updates

- If new models are added or existing models are modified, update the `model_names` list in the `process_all_models` function.
- When updating coefficients or recalibrators, ensure that the new files maintain the same structure and naming conventions as the original files.

For any questions or issues during implementation, please contact the system administrator or the development team.
