import random
import uuid
import time
from faker import Faker

fake = Faker()

def generate_transaction():
    """
    Generate a synthetic transaction with the fields expected by the model:
    trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender,
    street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time,
    merch_lat, merch_long.
    """

    trans_datetime = fake.date_time_between(start_date='-1y', end_date='now')
    trans_date_trans_time = trans_datetime.strftime('%Y-%m-%d %H:%M:%S')
    unix_time = int(time.mktime(trans_datetime.timetuple()))
    first_name = fake.first_name()
    last_name = fake.last_name()
    gender = random.choice(['M', 'F'])
    street = fake.street_address()
    city = fake.city()
    state = fake.state_abbr()
    zip_code = fake.zipcode()
    lat = float(fake.latitude())
    lng = float(fake.longitude())
    city_pop = random.randint(1000, 1000000)
    job = fake.job()
    dob_dt = fake.date_of_birth(minimum_age=18, maximum_age=90)
    dob_str = dob_dt.strftime('%Y-%m-%d')
    cc_num = fake.credit_card_number()
    possible_merchants = [
        'fraud_Kihn_Abernathy_and_Douglas',
        'shop_Barton_Bins',
        'store_Jones_Inc',
        'misc_net',
        'grocery_pos',
        'entertainment'
    ]
    merchant = random.choice(possible_merchants)
    possible_categories = [
        'shopping_net',
        'grocery_pos',
        'entertainment',
        'misc_net',
        'misc_pos'
    ]
    category = random.choice(possible_categories)
    amt = round(random.uniform(1.0, 2000.0), 2)
    trans_num = str(uuid.uuid4())
    merch_lat = lat + random.uniform(-0.05, 0.05)
    merch_long = lng + random.uniform(-0.05, 0.05)

    transaction = {
        "trans_date_trans_time": trans_date_trans_time,
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "first": first_name,
        "last": last_name,
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": lng,
        "city_pop": city_pop,
        "job": job,
        "dob": dob_str,
        "trans_num": trans_num,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long
    }

    return transaction
