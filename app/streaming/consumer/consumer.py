import json
from ..config import create_kafka_consumer, get_redis_connection
import logging
import sys
import os

# Add project root to Python path to find scripts module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from scripts.feature_engineering import MissingValueHandler, FraudFeatureTransformer
from scripts.predict import predict_transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = get_redis_connection()

# Initialize Redis keys if not already set
if not r.exists("total_transactions"):
    r.set("total_transactions", 0)
if not r.exists("total_fraud_transactions"):
    r.set("total_fraud_transactions", 0)

LAST_10_KEY = "last_10_transactions"
TRANSACTIONS_HISTORY_KEY = "transactions_history"

def detect_fraud(transaction):
    """
    Use the trained machine learning model to detect fraud.
    Return transaction with fraud status and probability.
    """
    try:
        result = predict_transaction(transaction)
        transaction["fraud_status"] = "FRAUD DETECTED🚨" if result['is_fraud'] else "NO FRAUD DETECTED ✅"
        transaction["fraud_probability"] = result['fraud_probability']
        logger.info(f"Prediction: {result['is_fraud']}, Probability: {result['fraud_probability']:.4f}")
        return transaction
    
    except Exception as e:
        logger.error(f"Error predicting fraud: {e}")
        # Fallback to random fraud detection instead of returning None
        transaction["fraud_status"] = "NOT PROCESSED"
        transaction["fraud_probability"] = 0.0
        logger.info(f"Using fallback prediction: {transaction['fraud_status']}")
        return transaction  # Always return the transaction object

def update_redis(transaction):
    """
    Update Redis with the new transaction:
      - Increment the total transactions counter.
      - Push the updated transaction (with fraud flag) onto a list.
      - Trim the list to the last 10 transactions.
    """
        
    is_fraud = transaction.get("fraud_status", "").startswith("FRAUD DETECTED")
    if is_fraud:
        r.incr("total_fraud_transactions")

    r.incr("total_transactions")
    
    r.lpush(LAST_10_KEY, json.dumps(transaction))
    r.ltrim(LAST_10_KEY, 0, 9)

    r.lpush(TRANSACTIONS_HISTORY_KEY, json.dumps(transaction))
    r.ltrim(TRANSACTIONS_HISTORY_KEY, 0, 99)

consumer = create_kafka_consumer("transactions")

for message in consumer:
    transaction = message.value
    print(f"Received transaction: {transaction}")
    if transaction:  # Make sure transaction is not None
        processed_transaction = detect_fraud(transaction)
        if processed_transaction:
            logger.info(f"Processed transaction: {processed_transaction}")
            update_redis(processed_transaction)
        else:
            logger.error("Failed to process transaction, detect_fraud returned None")
    else:
        logger.error("Received empty transaction")
