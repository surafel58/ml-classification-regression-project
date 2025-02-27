import streamlit as st
import redis
import json
import pandas as pd

# -----------------------------
# Helper Functions
# -----------------------------

def get_redis_connection():
    """Establish and return a connection to Redis."""
    return redis.Redis(host='localhost', port=6379, db=0)

def fetch_total_transactions(r):
    """Retrieve the total transaction count from Redis."""
    total = r.get("total_transactions")
    return int(total) if total else 0

def fetch_fraud_transactions(r):
    """Retrieve the fraudulent transaction count from Redis."""
    fraud = r.get("total_fraud_transactions")
    return int(fraud) if fraud else 0

def fetch_last_transactions(r, limit=10):
    """Retrieve the last `limit` transactions from Redis and convert them to a DataFrame."""
    transactions = r.lrange("last_10_transactions", 0, limit - 1)
    if transactions:
        # Convert each JSON string to a Python dict
        transactions = [json.loads(tx) for tx in transactions]
        return pd.DataFrame(transactions)
    return pd.DataFrame()

def display_metrics(total, fraud):
    """Display the total and fraudulent transaction counts as metrics."""
    col1, col2 = st.columns(2)
    col1.metric(label="Total Processed Transactions", value=total)
    col2.metric(label="Fraudulent Transactions", value=fraud)

def display_transactions_table(df):
    """Display the last 10 transactions as a table."""
    st.subheader("Last 10 Transactions")
    if not df.empty:
        st.dataframe(df)
    else:
        st.write("No transactions available yet.")

def update_dashboard():
    """Fetch data from Redis and update the dashboard display."""
    r = get_redis_connection()
    total = fetch_total_transactions(r)
    fraud = fetch_fraud_transactions(r)
    df = fetch_last_transactions(r)
    
    st.subheader("Metrics")
    display_metrics(total, fraud)
    display_transactions_table(df)

# -----------------------------
# Real-Time Fragment for Live Updates
# -----------------------------

# The fragment will refresh only this portion of the UI every 2 seconds.
@st.fragment(run_every=2)
def dashboard_fragment():
    update_dashboard()

# -----------------------------
# Main Dashboard Logic
# -----------------------------

def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
    st.title("Real-Time Fraud Detection Dashboard")
    
    # Call the fragment. Only this fragment will refresh periodically.
    dashboard_fragment()

if __name__ == "__main__":
    main()
