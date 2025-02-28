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
        # Sort categories to ensure consistent order
        self.all_categories = sorted(['grocery_pos', 'misc_net', 'others',
                                    'shopping_net', 'shopping_pos'])
                                    # , 'gas_transport'])

        # Define the exact order of columns that will be output
        self.output_columns = ['amt', 'gender', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long'] + [f'category_{cat}' for cat in self.all_categories] + ['hour', 'day', 'month', 'day_of_week', 'is_night', 'is_weekend', 'is_high_amount', 'risk_score']

    def fit(self, X, y=None):
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

        # Handle categories - vectorized operations
        X['category'] = X['category'].apply(
            lambda x: x if x in IMPORTANT_CATEGORIES else 'others'
        )

        # Initialize category columns with zeros
        for cat in self.all_categories:
            X[f'category_{cat}'] = 0

        # Set category values using vectorized operation
        for cat in self.all_categories:
            X.loc[X['category'] == cat, f'category_{cat}'] = 1

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

        # Ensure columns are in the correct order
        missing_cols = set(self.output_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0

        return X[self.output_columns]