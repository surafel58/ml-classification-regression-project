import time
from .data_generator import generate_transaction
import random
from ..config import create_kafka_producer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

topic_name = 'transactions'
producer = create_kafka_producer()

def produce_transactions():
    while True:
        transaction = generate_transaction()
        producer.send(topic_name, transaction)
        producer.flush()  # ensures message is sent immediately
        logger.info(f"Sent: {transaction}")
        time.sleep(random.uniform(1, 2))

if __name__ == "__main__":
    produce_transactions()
