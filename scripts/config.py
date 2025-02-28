"""
Configuration settings for the fraud detection pipeline.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = os.getcwd()

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')

# Model artifacts
PIPELINE_PATH = os.path.join(MODEL_DIR, 'fraud_detection_pipeline.joblib')
# SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Feature settings
IMPORTANT_CATEGORIES = [
    'grocery_pos', 
    'shopping_net', 
    'misc_net',
    'shopping_pos', 
    'gas_transport'
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Threshold settings
HIGH_AMOUNT_THRESHOLD = 200.0
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 5

# Risk score weights
RISK_WEIGHTS = {
    'is_night': 0.2,
    'is_weekend': 0.1,
    'is_high_amount': 0.3
} 