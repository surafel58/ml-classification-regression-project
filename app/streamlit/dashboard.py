from app.streaming.config import get_redis_connection
from app.streamlit.utils import (
    fetch_total_transactions,
    fetch_fraud_transactions,
    display_metrics,
    display_transactions_table,
    plot_transactions_over_time,
    plot_fraud_by_category,
    plot_transaction_amount_distribution_split,
)
import streamlit as st
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_dashboard(r):
    """Fetch data from Redis and update the dashboard display."""
    total = fetch_total_transactions(r)
    fraud = fetch_fraud_transactions(r)

    display_metrics(total, fraud)

    display_transactions_table(r)

    plot_transactions_over_time(r)

    plot_fraud_by_category(r)

    plot_transaction_amount_distribution_split(r)

# This fragment refreshes every 2 seconds
@st.fragment(run_every=2)
def dashboard_fragment(r):
    try:
        update_dashboard(r)
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        st.error("Error updating dashboard. Check logs for details.")


def main():
    r = get_redis_connection()
    st.markdown(body="# Real-Time Dashboard", help="""
    This dashboard presents real-time fraud detection data from our streaming pipeline:
    
    - Total Processed Transactions: The cumulative count of all transactions processed by our system
    - Fraudulent Transactions: The number of transactions flagged as potentially fraudulent
    
    The data below shows the most recent transactions, their details, and fraud status. Our ML model 
    analyzes each transaction in real-time and assigns a fraud probability score.
    
    The visualizations help identify patterns in fraudulent activity across different categories,
    time periods, and transaction amounts.
    """)

    dashboard_fragment(r)


main()
