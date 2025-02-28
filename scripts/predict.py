"""
Prediction script for the fraud detection model.
"""

import joblib
import pandas as pd
from scripts.config import PIPELINE_PATH

def load_pipeline():
    """Load the trained pipeline."""
    return joblib.load(PIPELINE_PATH)

def predict_transaction(transaction_data, pipeline=None):
    """
    Predict whether a transaction is fraudulent.

    Args:
        transaction_data (dict or pd.DataFrame): Transaction data to predict
        pipeline (Pipeline, optional): Loaded pipeline. If None, will load from disk

    Returns:
        dict: Prediction results containing fraud probability and binary prediction
    """
    # Load pipeline if not provided
    if pipeline is None:
        pipeline = load_pipeline()

    # Convert dictionary to DataFrame if needed
    if isinstance(transaction_data, dict):
        transaction_data = pd.DataFrame([transaction_data])

    # Make prediction
    prediction = pipeline.predict(transaction_data)
    probability = pipeline.predict_proba(transaction_data)[:, 1]

    return {
        'is_fraud': bool(prediction[0]),
        'fraud_probability': float(probability[0])
    }

def predict_batch(transactions, pipeline=None):
    """
    Predict fraud for multiple transactions.

    Args:
        transactions (pd.DataFrame): DataFrame containing multiple transactions
        pipeline (Pipeline, optional): Loaded pipeline. If None, will load from disk

    Returns:
        pd.DataFrame: Original data with predictions added
    """
    # Load pipeline if not provided
    if pipeline is None:
        pipeline = load_pipeline()

    # Make predictions
    predictions = pipeline.predict(transactions)
    probabilities = pipeline.predict_proba(transactions)[:, 1]

    # Add predictions to DataFrame
    results = transactions.copy()
    results['is_fraud_predicted'] = predictions
    results['fraud_probability'] = probabilities

    return results

if __name__ == "__main__":
    # Example usage
    example_transaction = {
        'trans_date_trans_time': '2023-01-01 02:45:00',
        'amt': 999.99,
        'category': 'shopping_net',
        'gender': 'M',
    }

    result = predict_transaction(example_transaction)
    print("\nPrediction for example transaction:")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")