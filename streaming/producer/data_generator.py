from faker import Faker
import random

fake = Faker()

def generate_transaction():
    """Generate a fake credit card transaction."""
    transaction = {
        "transaction_id": fake.uuid4(),
        "timestamp": fake.iso8601(tzinfo=None, end_datetime=None),
        "amount": round(random.uniform(1.0, 1000.0), 2),
        "card_number": fake.credit_card_number(),
        "merchant": fake.company(),
        "location": fake.city(),
        "status": random.choice(["approved", "declined"])
    }
    return transaction