#!/usr/bin/env python3
"""
Test script to verify the XGBoost to Core ML conversion.
This script:
1. Runs the conversion
2. Loads both models (XGBoost and Core ML)
3. Makes predictions with sample data
4. Compares the results
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import coremltools as ct

# Import the conversion script
from convert_xgboost_to_coreml import load_xgboost_model, load_scaler, load_feature_importance

def test_conversion():
    """Test the conversion and verify predictions"""
    print("Testing XGBoost to Core ML conversion...")
    
    # File paths
    model_path = 'model/mental_health_model.xgb'
    scaler_path = 'model/scaler.save'
    feature_importance_path = 'model/feature_importance.csv'
    coreml_path = 'model/mental_health_model.mlmodel'
    
    # Check if conversion script has been run
    if not os.path.exists(coreml_path):
        print("Core ML model not found. Running conversion script...")
        import subprocess
        result = subprocess.run(['python', 'convert_xgboost_to_coreml.py'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running conversion script: {result.stderr}")
            sys.exit(1)
    
    # Load XGBoost model
    xgb_model = load_xgboost_model(model_path)
    
    # Load scaler
    scaler = load_scaler(scaler_path)
    
    # Load feature importance
    feature_importance = load_feature_importance(feature_importance_path)
    
    # Get feature names
    feature_names = list(feature_importance.keys())
    
    # Create sample data (random values within reasonable ranges for each feature)
    np.random.seed(42)  # For reproducibility
    
    # Create a dictionary with reasonable sample values for each feature
    sample_data = {
        'activity_very_active_minutes_mean': 30.0,
        'sleep_minute_light_std': 45.2,
        'sleep_minute_rem_std': 15.6,
        'sleep_minute_asleep_std': 22.3,
        'activity_very_active_minutes_std': 12.5,
        'activity_sedentary_minutes_std': 35.7,
        'activity_steps_max': 15000.0,
        'activity_steps_mean': 8500.0,
        'age': 35.0,
        'activity_steps_std': 2500.0,
        'hr_avg_rate_min': 55.0,
        'activity_sedentary_minutes_mean': 480.0,
        'hr_avg_rate_mean': 72.0,
        'sleep_social_jetlag_': 1.2,
        'sleep_minute_rem_mean': 95.0,
        'sleep_minute_awake_mean': 30.0,
        'sleep_minute_light_mean': 240.0,
        'sleep_minute_asleep_max': 480.0,
        'activity_calories_out_mean': 2200.0,
        'sleep_time_diff_mean': 0.5,
        'activity_calories_out_std': 300.0,
        'activity_activity_ratio_mean': 0.3,
        'hr_avg_rate_skew': 0.2,
        'activity_fairly_active_minutes_mean': 45.0,
        'activity_lightly_active_minutes_mean': 180.0,
        'hr_avg_rate_max': 120.0,
        'activity_fairly_active_minutes_std': 15.0,
        'sleep_time_diff_std': 0.3,
        'activity_very_active_minutes_max': 60.0,
        'sleep_minute_deep_mean': 90.0,
        'activity_lightly_active_minutes_std': 30.0,
        'sleep_minute_asleep_mean': 420.0,
        'sleep_minute_deep_std': 12.0,
        'sleep_minute_asleep_min': 360.0,
        'sleep_minute_awake_std': 10.0,
        'activity_activity_ratio_std': 0.05,
        'hr_avg_rate_std': 8.0,
        'gender_binary': 1.0
    }
    
    # Convert to numpy array in the right order for XGBoost
    sample_array = np.array([sample_data[feature] for feature in feature_names]).reshape(1, -1)
    
    # Apply scaling if scaler is available
    if scaler is not None:
        scaled_array = scaler.transform(sample_array)
    else:
        scaled_array = sample_array
    
    # Prepare DMatrix for XGBoost prediction
    dmatrix = xgb.DMatrix(scaled_array, feature_names=feature_names)
    
    # Get XGBoost prediction
    xgb_prediction = xgb_model.predict(dmatrix)
    print(f"XGBoost prediction: {xgb_prediction}")
    
    # Load Core ML model
    coreml_model = ct.models.MLModel(coreml_path)
    
    # Get model type from metadata
    model_type = coreml_model.user_defined_metadata.get('model_type', 'classifier')
    
    # Prepare input for Core ML prediction (no need to scale - should be handled by the model)
    coreml_input = {}
    for feature in feature_names:
        coreml_input[feature] = sample_data[feature]
    
    # Make Core ML prediction
    coreml_prediction = coreml_model.predict(coreml_input)
    print(f"Core ML prediction: {coreml_prediction}")
    
    # Check if predictions match
    # For classifiers, compare prediction labels
    # For regressors, check if values are close
    if model_type == 'classifier':
        # For binary classifier, XGBoost outputs probabilities but Core ML outputs labels
        # Convert XGBoost probability to class index for comparison
        if len(xgb_prediction.shape) == 1:  # Binary classification
            xgb_class = 1 if xgb_prediction[0] >= 0.5 else 0
            match = str(xgb_class) == str(coreml_prediction['prediction'])
        else:  # Multi-class
            xgb_class = np.argmax(xgb_prediction[0])
            match = str(xgb_class) == str(coreml_prediction['prediction'])
    else:  # Regressor
        match = np.isclose(xgb_prediction[0], coreml_prediction['prediction'], rtol=1e-3)
    
    if match:
        print("✅ Test passed: XGBoost and Core ML predictions match!")
    else:
        print("❌ Test failed: Predictions do not match.")
        print(f"XGBoost prediction type: {type(xgb_prediction)}, shape: {xgb_prediction.shape}")
        print(f"Core ML prediction type: {type(coreml_prediction['prediction'])}")
    
    # Print model metadata
    print("\nCore ML Model Information:")
    print(f"Author: {coreml_model.author}")
    print(f"License: {coreml_model.license}")
    print(f"Description: {coreml_model.short_description}")
    print(f"Input features: {len(coreml_model.input_description._fd_)}")
    print(f"Output features: {len(coreml_model.output_description._fd_)}")
    
    # Check for scaling parameters in metadata
    if 'scaling_params' in coreml_model.user_defined_metadata:
        print("\nScaling parameters found in metadata")
    else:
        print("\nNo scaling parameters found in metadata")
    
    # Check for feature importance in metadata
    if 'feature_importance' in coreml_model.user_defined_metadata:
        print("Feature importance found in metadata")
    else:
        print("No feature importance found in metadata")
    
    return match

if __name__ == "__main__":
    test_conversion() 