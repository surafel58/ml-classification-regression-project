from kafka import KafkaProducer
import json
import time
from data_generator import generate_transaction
import random
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = 'transactions'

def produce_transactions():
    while True:
        transaction = generate_transaction()
        producer.send(topic_name, transaction)
        producer.flush()  # ensures message is sent immediately
        print(f"Sent: {transaction}")
        time.sleep(random.uniform(0.1, 2.0))  # random delay between 0.1-2 seconds to simulate real transaction patterns

if __name__ == "__main__":
    produce_transactions()
