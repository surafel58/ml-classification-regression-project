import json
import random
from ..config import create_kafka_consumer, get_redis_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = get_redis_connection()

# Initialize Redis keys if not already set
if not r.exists("total_transactions"):
    r.set("total_transactions", 0)
if not r.exists("total_fraud_transactions"):
    r.set("total_fraud_transactions", 0)

LAST_10_KEY = "last_10_transactions"

def simulate_fraud_detection(transaction):
    """
    Simulate fraud detection by randomly flagging a transaction as fraudulent.
    In your final model, you'll replace this logic with an actual inference call.
    """
    # For demonstration, mark 10% of transactions as fraud
    is_fraud = random.random() < 0.1
    transaction["fraud_status"] = "FRAUD DETECTED" if is_fraud else "NO FRAUD DETECTED"
    return transaction

def update_redis(transaction):
    """
    Update Redis with the new transaction:
      - Increment the total transactions counter.
      - Push the updated transaction (with fraud flag) onto a list.
      - Trim the list to the last 10 transactions.
    """
    is_fraud = transaction["fraud_status"] == "FRAUD DETECTED"
    if is_fraud:
        r.incr("total_fraud_transactions")

    r.incr("total_transactions")
    
    r.lpush(LAST_10_KEY, json.dumps(transaction))
    r.ltrim(LAST_10_KEY, 0, 9)

consumer = create_kafka_consumer("transactions")

for message in consumer:
    transaction = message.value
    transaction = simulate_fraud_detection(transaction)
    logger.info(f"Processed transaction: {transaction}")
    update_redis(transaction)
