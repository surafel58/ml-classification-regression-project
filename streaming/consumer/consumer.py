import json
import random
from kafka import KafkaConsumer
import redis

# Connect to Redis (ensure Redis is running)
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Redis keys if not already set
if not r.exists("total_transactions"):
    r.set("total_transactions", 0)

LAST_10_KEY = "last_10_transactions"

def simulate_fraud_detection(transaction):
    """
    Simulate fraud detection by randomly flagging a transaction as fraudulent.
    In your final model, you'll replace this logic with an actual inference call.
    """
    # For demonstration, mark 10% of transactions as fraud
    transaction["is_fraud"] = random.random() < 0.1
    return transaction

def update_redis(transaction):
    """
    Update Redis with the new transaction:
      - Increment the total transactions counter.
      - Push the updated transaction (with fraud flag) onto a list.
      - Trim the list to the last 10 transactions.
    """
    r.incr("total_transactions")
    r.lpush(LAST_10_KEY, json.dumps(transaction))
    r.ltrim(LAST_10_KEY, 0, 9)

# Create Kafka consumer for the 'transactions' topic
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    # auto_offset_reset='earliest',
    # enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    transaction = message.value
    # Simulate fraud detection by appending an "is_fraud" field
    transaction = simulate_fraud_detection(transaction)
    print("Processed transaction:", transaction)
    update_redis(transaction)
