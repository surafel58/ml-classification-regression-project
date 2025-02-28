"""
Script to evaluate the trained model on the test dataset.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.config import (
    DATA_DIR, MODEL_DIR, PIPELINE_PATH
)

def load_test_data():
    """Load the test dataset."""
    useful_columns = [
        'trans_date_trans_time', 'category', 'amt',
        'gender', 'lat', 'long', 'city_pop',
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    return pd.read_csv(os.path.join(DATA_DIR, "fraudTest.csv"), usecols=useful_columns)

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Evaluate model performance and print metrics."""
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Calculate and print ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    return {
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_results(y_true, results):
    """Plot ROC and Precision-Recall curves."""
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_true, results['y_pred_proba'])
    plt.figure(figsize=(10, 5))

    # Plot ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, results['y_pred_proba'])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()

def analyze_feature_importance(pipeline, X_test):
    """Analyze and plot feature importance."""
    # Get feature importance from the model
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    feature_names = X_test.columns

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.show()

    return importance_df

def main():
    """Main evaluation function."""
    # Load the trained pipeline
    print("Loading trained model...")
    pipeline = joblib.load(PIPELINE_PATH)

    # Load test data
    print("Loading test data...")
    test_data = load_test_data()

    # Split features and target
    X_test = test_data.drop(['is_fraud'], axis=1)
    y_test = test_data['is_fraud']

    # Make predictions
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate model
    print("Evaluating model performance...")
    results = evaluate_model(y_test, y_pred, y_pred_proba)

    # Plot results
    print("\nGenerating performance plots...")
    plot_results(y_test, results)

    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    X_test_transformed = pipeline.named_steps['feature_engineer'].transform(X_test)
    importance_df = analyze_feature_importance(pipeline, X_test_transformed)

    # Print top 10 most important features
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

if __name__ == "__main__":
    main()