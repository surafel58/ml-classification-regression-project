import streamlit as st
import json
import pandas as pd
import sys
import os
import logging

# Make sure the project root is on sys.path so "app" can be imported as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.predict import predict_transaction
from app.streaming.config import get_redis_connection

#############################
# Existing Real-Time Dashboard Code
#############################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_total_transactions(r):
    total = r.get("total_transactions")
    return int(total) if total else 0

def fetch_fraud_transactions(r):
    fraud = r.get("total_fraud_transactions")
    return int(fraud) if fraud else 0

def fetch_last_transactions(r, limit=10):
    transactions = r.lrange("last_10_transactions", 0, limit - 1)
    if transactions:
        transactions = [json.loads(tx) for tx in transactions]
        return pd.DataFrame(transactions)
    return pd.DataFrame()

def display_metrics(total, fraud):
    col1, col2 = st.columns(2)
    col1.metric(label="Total Processed Transactions", value=total)
    col2.metric(label="Fraudulent Transactions", value=fraud)

def display_transactions_table(df):
    st.subheader("Last 10 Transactions")
    if not df.empty:
        st.dataframe(df)
    else:
        st.write("No transactions available yet.")


def plot_transactions_over_time(r):
    """
    Fetch the last 100 transactions from Redis, group them by minute,
    and display a line chart of transaction count over time.
    """
    transactions = r.lrange('last_100_transactions', 0, 99)
    if transactions:
        transactions = [json.loads(tx) for tx in transactions]
        df = pd.DataFrame(transactions)
        
        # Ensure the timestamp column exists and is converted to datetime.
        if "trans_date_trans_time" in df.columns:
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
            
            # Group by minute (or any desired time interval) and count transactions.
            df_grouped = (
                df.groupby(pd.Grouper(key="trans_date_trans_time", freq="T"))
                  .size()
                  .reset_index(name="transaction_count")
            )
            
            st.subheader("Transactions Over Time")
            st.line_chart(df_grouped.set_index("trans_date_trans_time"))
        else:
            st.write("No valid timestamp field in the data.")
    else:
        st.write("No transaction history data available.")


def update_dashboard(r):
    """Fetch data from Redis and update the dashboard display."""
    total = fetch_total_transactions(r)
    fraud = fetch_fraud_transactions(r)
    df = fetch_last_transactions(r)
    st.subheader("Metrics")
    display_metrics(total, fraud)
    display_transactions_table(df)
    # plot_transactions_over_time(r)

# This fragment refreshes every 2 seconds
@st.fragment(run_every=2)
def dashboard_fragment(r):
    try:
        update_dashboard(r)
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        st.error("Error updating dashboard. Check logs for details.")

#############################
# Manual Data Entry & Prediction
#############################

def detect_fraud_manual_input(input_data: dict) -> dict:
    """
    Example placeholder function to 'predict' fraud.
    In practice, you'd load your model/pipeline and run a real prediction.
    Returns a dict with 'fraud_status' and 'fraud_probability'.
    """
    # integrate our model here
    result = predict_transaction(input_data)

    return {
        "fraud_status": "FRAUD DETECTED" if result['is_fraud'] else "NO FRAUD DETECTED",
        "fraud_probability": result['fraud_probability']
    }

def add_data_manually():
    st.subheader("Add Data Manually for Fraud Prediction")

    # Example fields – adapt to match your model’s actual columns
    trans_date_time = st.text_input("Transaction Date Time (YYYY-MM-DD HH:MM:SS)", "2019-01-01 00:00:00")
    cc_num = st.text_input("Credit Card Number", "1234567890123456")
    merchant = st.text_input("Merchant", "fraud_Kihn_Abernathy_and_Douglas")
    category = st.selectbox("Category", ["shopping_net", "grocery_pos", "entertainment", "misc_net", "misc_pos"])
    amt = st.number_input("Amount", min_value=0.0, max_value=9999.99, value=25.0, step=0.01)
    first_name = st.text_input("First Name", "John")
    last_name = st.text_input("Last Name", "Doe")
    gender = st.selectbox("Gender", ["M", "F"])
    street = st.text_input("Street", "123 Main St")
    city = st.text_input("City", "ExampleCity")
    state = st.text_input("State", "CA")
    zip_code = st.text_input("Zip", "12345")
    lat = st.number_input("Latitude", value=37.7749, step=0.0001)
    long_ = st.number_input("Longitude", value=-122.4194, step=0.0001)
    city_pop = st.number_input("City Population", value=100000, step=1)
    job = st.text_input("Job", "Data Scientist")
    dob = st.text_input("DOB (YYYY-MM-DD)", "1990-01-01")
    trans_num = st.text_input("Transaction Number", "abc123xyz")
    unix_time = st.number_input("Unix Time", value=1546300800, step=1)
    merch_lat = st.number_input("Merchant Latitude", value=37.7749, step=0.0001)
    merch_long = st.number_input("Merchant Longitude", value=-122.4194, step=0.0001)

    # Submit button
    if st.button("Predict Fraud"):
        input_data = {
            "trans_date_trans_time": trans_date_time,
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
            "long": long_,
            "city_pop": city_pop,
            "job": job,
            "dob": dob,
            "trans_num": trans_num,
            "unix_time": unix_time,
            "merch_lat": merch_lat,
            "merch_long": merch_long
        }

        # Here we call the placeholder detect_fraud function
        result = detect_fraud_manual_input(input_data)

        st.write("## Prediction Result")
        st.write(f"**Fraud Status:** {result['fraud_status']}")
        st.write(f"**Fraud Probability:** {result['fraud_probability']}")

#############################
# Main Streamlit App
#############################

def main():
    r = get_redis_connection()
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
    st.title("Fraud Detection Dashboard")

    # Sidebar navigation
    menu_option = st.sidebar.radio(
        "Navigation",
        ["View Real-Time Data", "Add Data Manually"]
    )

    if menu_option == "View Real-Time Data":
        # Show the existing real-time data fragment
        dashboard_fragment(r)
    elif menu_option == "Add Data Manually":
        # Show a form to manually input data for fraud detection
        add_data_manually()

if __name__ == "__main__":
    main()
