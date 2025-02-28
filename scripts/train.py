"""
Training script for the fraud detection model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

from scripts.config import (
    DATA_DIR, MODEL_DIR, PIPELINE_PATH,
    RANDOM_STATE, TEST_SIZE
)
from scripts.feature_engineering import MissingValueHandler, FraudFeatureTransformer

def load_data(data_path):
    """Load the fraud detection dataset."""
    return pd.read_csv(data_path)

def create_pipeline():
    """Create the fraud detection pipeline."""
    return Pipeline(steps=[
        ('missing_handler', MissingValueHandler()),
        ('feature_engineer', FraudFeatureTransformer()),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist'
        ))
    ], memory='./cache')

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")


def main():
    """Main training function."""
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    data_path = os.path.join(DATA_DIR, "fraudTrain.csv")
    df = load_data(data_path)

    # Split features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Create and train pipeline
    print("Training model...")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(pipeline, X_test, y_test)

    # Save pipeline
    print(f"Saving pipeline to {PIPELINE_PATH}...")
    joblib.dump(pipeline, PIPELINE_PATH)
    print("Training completed!")

if __name__ == "__main__":
    main()