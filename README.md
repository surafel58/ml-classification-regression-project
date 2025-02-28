# Credit Card Fraud Detection

This project implements a machine learning pipeline for detecting fraudulent credit card transactions. The system uses XGBoost and feature engineering based on domain knowledge to identify potential fraud cases.

## Project Structure

```
├── dataset/               # Data directory
├── model/                 # Saved model artifacts
├── notebooks/            # Jupyter notebooks for EDA
├── scripts/              # Python scripts
│   ├── config.py         # Configuration settings
│   ├── feature_engineering.py  # Feature transformers
│   ├── train.py          # Model training script
│   └── predict.py        # Prediction script
└── requirements.txt      # Project dependencies
```

## Key Features

- Handles imbalanced dataset (0.58% fraud vs 99.42% non-fraud)
- Feature engineering based on domain knowledge:
  - Time-based features (hour, day, month)
  - High-risk flags (night transactions, weekends)
  - Amount-based features
  - Category encoding
- Risk scoring system
- Production-ready pipeline with scikit-learn transformers

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:

```bash
python scripts/train.py
```

This will:
1. Load and preprocess the data
2. Train the model
3. Evaluate performance
4. Save the pipeline to `model/fraud_detection_pipeline.joblib`

### Making Predictions

```python
from scripts.predict import predict_transaction

# Single transaction
transaction = {
    'trans_date_trans_time': '2023-01-01 02:45:00',
    'amt': 999.99,
    'category': 'shopping_net',
    'gender': 'M',
    'hour': 2
}

result = predict_transaction(transaction)
print(f"Is Fraud: {result['is_fraud']}")
print(f"Fraud Probability: {result['fraud_probability']}")
```

## Model Performance

The XGBoost model achieves:
- ROC AUC Score: 0.9955
- F1 Score: 0.8166
- High precision in fraud detection while maintaining good recall

## Key Insights

1. Fraud patterns:
   - More frequent in transactions over $200
   - Higher risk during night hours
   - More common on weekends and Mondays
   - Peak months: April, June, January, February

2. High-risk categories:
   - grocery_pos
   - shopping_net
   - misc_net
   - shopping_pos
   - gas_transport

## Contributing

Feel free to submit issues and enhancement requests!
