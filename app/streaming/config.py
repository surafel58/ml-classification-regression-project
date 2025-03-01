import os
import json
import redis
from kafka import KafkaProducer, KafkaConsumer
import logging
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment Variable Constants
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

print(f"environment variables: {KAFKA_BOOTSTRAP_SERVERS}, {REDIS_HOST}, {REDIS_PORT}, {REDIS_DB}")
# Kafka Configuration
KAFKA_CONFIG = {
    "bootstrap_servers": [KAFKA_BOOTSTRAP_SERVERS],
    "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
}

# Redis Connection Helper
def get_redis_connection():
    """Establish and return a connection to Redis."""
    try:
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        raise e

# Kafka Producer Helper
def create_kafka_producer():
    """Create and return a KafkaProducer using the configuration."""
    try:
        return KafkaProducer(**KAFKA_CONFIG)
    except Exception as e:
        logger.error(f"Error creating Kafka producer: {e}")
        raise e

# Kafka Consumer Helper
def create_kafka_consumer(topic, group_id=None, auto_offset_reset="earliest"):
    """
    Create and return a KafkaConsumer for a given topic.
    
    :param topic: The topic name to subscribe to.
    :param group_id: Optional consumer group id.
    :param auto_offset_reset: Where to start reading if no offset is committed.
    """
    try:
        return KafkaConsumer(
            topic,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
            value_deserializer=lambda m: json.loads(m.decode("utf-8"))
        )
    except Exception as e:
        logger.error(f"Error creating Kafka consumer: {e}")
