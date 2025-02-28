"""
Feature engineering transformers for the fraud detection pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scripts.config import (
    IMPORTANT_CATEGORIES, HIGH_AMOUNT_THRESHOLD,
    NIGHT_START_HOUR, NIGHT_END_HOUR, RISK_WEIGHTS
)

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handles missing values in the dataset."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Fill numeric columns with median
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = X.select_dtypes(exclude=[np.number]).columns
        X[categorical_columns] = X[categorical_columns].fillna(X[categorical_columns].mode().iloc[0])
        
        return X

class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Creates fraud detection features based on domain knowledge."""
    
    def __init__(self):
        self.all_categories = IMPORTANT_CATEGORIES + ['others']
    
    def fit(self, X, y=None):
        # Nothing to learn, but we need this for the sklearn pipeline
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert transaction datetime
        X['datetime'] = pd.to_datetime(X['trans_date_trans_time'])
        
        # Extract time-based features
        X['hour'] = X['datetime'].dt.hour
        X['day_of_week'] = X['datetime'].dt.dayofweek
        X['month'] = X['datetime'].dt.month
        
        # Create binary flags
        X['is_night'] = ((X['hour'] >= NIGHT_START_HOUR) | 
                        (X['hour'] <= NIGHT_END_HOUR)).astype(int)
        X['is_weekend'] = (X['day_of_week'].isin([5, 6])).astype(int)
        X['is_high_amount'] = (X['amt'] > HIGH_AMOUNT_THRESHOLD).astype(int)
        
        # Handle categories
        X['category'] = X['category'].apply(
            lambda x: x if x in IMPORTANT_CATEGORIES else 'others'
        )
        
        # Create all category columns with zeros
        for cat in self.all_categories:
            X[f'category_{cat}'] = 0
        
        # Set 1 for the present categories
        for idx, cat in enumerate(X['category']):
            X.at[idx, f'category_{cat}'] = 1
        
        # Encode gender (M=1, F=0)
        if 'gender' in X.columns:
            X['gender'] = (X['gender'] == 'M').astype(int)
        
        # Calculate risk score
        X['risk_score'] = (
            X['is_night'] * RISK_WEIGHTS['is_night'] +
            X['is_weekend'] * RISK_WEIGHTS['is_weekend'] +
            X['is_high_amount'] * RISK_WEIGHTS['is_high_amount']
        )
        
        # Drop unnecessary columns
        columns_to_drop = [
            'trans_date_trans_time', 'datetime', 'category',
            'cc_num', 'merchant', 'first', 'last', 'street',
            'city', 'state', 'zip', 'job', 'dob', 'trans_num',
            'unix_time'
        ]
        
        # Only drop columns that exist
        columns_to_drop = [col for col in columns_to_drop if col in X.columns]
        X = X.drop(columns=columns_to_drop)
        
        return X 