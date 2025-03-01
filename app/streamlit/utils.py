import json
import numpy as np
import pandas as pd
import streamlit as st
from scripts.predict import predict_transaction

def fetch_total_transactions(r):
    total = r.get("total_transactions")
    return int(total) if total else 0

def fetch_fraud_transactions(r):
    fraud = r.get("total_fraud_transactions")
    return int(fraud) if fraud else 0

def fetch_transactions(r, key, limit=100):
    transactions = r.lrange(key, 0, limit - 1)
    if transactions:
        transactions = [json.loads(tx) for tx in transactions]
        return pd.DataFrame(transactions)
    return pd.DataFrame()

def display_metrics(total, fraud):
    col1, col2 = st.columns(2)
    col1.metric(label="Total Processed Transactions", value=total,border=True)
    col2.metric(label="Fraudulent Transactions", value=fraud,border=True)

def display_transactions_table(r):
    df = fetch_transactions(r, "last_10_transactions")
    ct = st.container(border=True)
    with ct:
        st.markdown("### Transactions History" ,help="These are the last 10 transactions processed by the system")
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No transactions available yet.")
    
def prepare_transactions_time_series(df):
    """
    Fetch the last 100 transactions from Redis (stored in "transactions_history"),
    convert timestamps to datetime, sort them, and group by minute to count transactions.
    
    Returns:
        A DataFrame with a datetime index and a column "transaction_count" representing
        the number of transactions per minute.
    """
    if df.empty:
        return pd.DataFrame(columns=["trans_date_trans_time", "transaction_count"])
    
    try:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"], 
            format='%Y-%m-%d %H:%M:%S',
            errors="coerce"
        )
        # Drop rows with invalid timestamps
        df = df.dropna(subset=["trans_date_trans_time"])
        
        # Restrict data to the last 60 minutes
        max_time = df["trans_date_trans_time"].max()

        df = df[df["trans_date_trans_time"] >= (max_time - pd.Timedelta(minutes=60))]
        df = df.sort_values("trans_date_trans_time")
        df = df.set_index("trans_date_trans_time")

        df_grouped = df.resample("T").size().reset_index(name="transaction_count")
        df_grouped = df_grouped.set_index("trans_date_trans_time")
        
        return df_grouped
    except Exception as e:
        st.error(f"Error preparing time series data: {e}")
        return pd.DataFrame(columns=["trans_date_trans_time", "transaction_count"])


def plot_transactions_over_time(r):
    """
    Fetch the last 100 transactions from Redis (stored in "transactions_history"),
    group them by minute, and display a line chart showing transaction counts over time.
    """
    df = fetch_transactions(r, "transactions_history")
    ct = st.container(border=True)
    with ct:
        st.markdown("### Transactions Over Time - Last 60 minutes", help="""
            The chart below shows the pattern of transactions over time, with fraudulent transactions highlighted.
            Unusual spikes or patterns may indicate coordinated fraud attempts.
            """)
        if df.empty:
            st.write("No transaction history available.")
            return

        try:
            chart_data = prepare_transactions_time_series(df)
            st.line_chart(chart_data, x_label="Time", y_label="Transactions")

        except Exception as e:
            st.error(f"Error plotting transactions over time: {e}")

def prepare_fraud_by_category_data(r):
    """
    Fetch transactions from Redis (using "transactions_history"),
    and group by the 'category' field to count how many transactions in each category
    are flagged as fraud (i.e., fraud_status == "FRAUD DETECTED").
    
    Returns:
        A DataFrame with columns 'category' and 'fraud_count'.
    """
    df = fetch_transactions(r, "transactions_history")
    if df.empty:
        return pd.DataFrame(columns=["category", "fraud_count"])
    
    try:
        # Create a column indicating fraud (1 if fraud, 0 otherwise)
        df["is_fraud"] = df["fraud_status"].apply(lambda x: 1 if x == "FRAUD DETECTED" else 0)

        # Map short category names to human-readable labels
        CATEGORY_LABELS = {
            "shopping_net": "Online Shopping",
            "grocery_pos": "Grocery (POS)",
            "entertainment": "Entertainment",
            "misc_net": "Misc (Online)",
            "misc_pos": "Misc (POS)"
        }

        df["category"] = df["category"].map(CATEGORY_LABELS).fillna(df["category"])

        # Group by category and sum the fraud counts
        df_grouped = df.groupby("category")["is_fraud"].sum().reset_index()
        df_grouped = df_grouped.rename(columns={"is_fraud": "fraud_count"})
        return df_grouped
    except Exception as e:
        st.error(f"Error preparing fraud-by-category data: {e}")
        return pd.DataFrame(columns=["category", "fraud_count"])

def plot_fraud_by_category(r):
    """
    Prepare data for the Fraud by Category chart and plot it as a bar chart.
    """
    df_cat = prepare_fraud_by_category_data(r)
    ct = st.container(border=True)
    with ct:
        st.markdown("### Fraud by Category", help="""
            This chart shows the distribution of fraudulent transactions across different categories.
            Categories with higher counts indicate areas where fraud is more prevalent.
            """)
        if df_cat.empty:
            st.write("No fraud data available.")
        else:
            # Set the 'category' column as the index for the bar chart
            df_cat = df_cat.set_index("category")
            st.bar_chart(df_cat, horizontal=True, height=400)


def bin_transaction_amounts_natural(df, bins=10):
    """
    Bin the transaction amounts in df['amt'] into the specified number of bins,
    round the bin edges to natural numbers, and sort the bins by the left edge.
    
    Returns a DataFrame with columns "Amount Range" and "Count".
    """
    if df.empty or "amt" not in df.columns:
        return pd.DataFrame(columns=["Amount Range", "Count"])
    
    # Convert 'amt' to numeric and drop any invalid values.
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df = df.dropna(subset=["amt"])
    if df.empty:
        return pd.DataFrame(columns=["Amount Range", "Count"])
    
    min_amt = df["amt"].min()
    max_amt = df["amt"].max()
    
    # Create evenly spaced bin edges and round them to natural numbers.
    edges = np.linspace(min_amt, max_amt, bins+1)
    edges = np.ceil(edges).astype(int)
    
    # Ensure edges are strictly increasing (handle edge-case if min == max)
    if len(np.unique(edges)) == 1:
        edges = np.array([edges[0], edges[0] + 1])
    
    # Bin the amounts.
    df["amount_bin"] = pd.cut(df["amt"], bins=edges, include_lowest=True)
    
    # Group by bin and count transactions.
    df_binned = df.groupby("amount_bin").size().reset_index(name="Count")
    
    # Sort bins by the left edge of the interval.
    df_binned = df_binned.sort_values(
        by="amount_bin", 
        key=lambda x: x.apply(lambda interval: interval.left)
    )
    
    # Create human-readable labels by rounding the edges.
    def format_interval(interval):
        if pd.isna(interval):
            return "N/A"
        left, right = int(interval.left), int(interval.right)
        return f"{left} - {right}"
    
    df_binned["Amount Range"] = df_binned["amount_bin"].apply(format_interval)
    
    return df_binned[["Amount Range", "Count"]]

def plot_transaction_amount_distribution_split(r, bins=10):
    """
    Prepare and display two horizontal bar charts side by side:
    one for fraudulent transactions and one for non-fraudulent transactions.
    Each chart shows the transaction amount distribution with human-readable, sorted ranges.
    """
    # Fetch the 100 most recent transactions
    df = fetch_transactions(r, "transactions_history")
    if df.empty:
        st.write("No transaction data available for amount distribution.")
        return

    # Ensure 'amt' is numeric and drop invalid values
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df = df.dropna(subset=["amt"])
    if df.empty:
        st.write("No valid transaction data available.")
        return

    # Create a column for fraud flag: 1 if fraud, else 0
    df["is_fraud"] = df["fraud_status"].apply(lambda x: 1 if x == "FRAUD DETECTED" else 0)

    # Split data into fraud and non-fraud subsets
    df_fraud = df[df["is_fraud"] == 1].copy()
    df_legit = df[df["is_fraud"] == 0].copy()

    # Use our natural binning function for each subset
    df_fraud_bins = bin_transaction_amounts_natural(df_fraud, bins=bins)
    df_legit_bins = bin_transaction_amounts_natural(df_legit, bins=bins)

    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2, border=True)

    with col1:
        st.markdown("### Fraudulent Transactions", help="""
        This chart shows the distribution of fraudulent transactions across different amount ranges.
        Amount ranges with higher counts indicate areas where fraud is more prevalent.
        """)
        if df_fraud_bins.empty:
            st.write("No fraudulent transaction data available.")
        else:
            # Set 'Amount Range' as index and plot horizontal bar chart
            df_fraud_bins = df_fraud_bins.set_index("Amount Range")
            st.bar_chart(df_fraud_bins, height=400, color=["#ff8800"])

    with col2:
        st.markdown("### Non-Fraudulent Transactions", help="""
        This chart shows the distribution of non-fraudulent transactions across different amount ranges.
        Amount ranges with higher counts indicate areas where non-fraudulent transactions are more prevalent.
        """)
        if df_legit_bins.empty:
            st.write("No non-fraudulent transaction data available.")
        else:
            df_legit_bins = df_legit_bins.set_index("Amount Range")
            st.bar_chart(df_legit_bins, height=400)


def detect_fraud_manual_input(input_data: dict) -> dict:
    """
    Example placeholder function to 'predict' fraud.
    In practice, you'd load your model/pipeline and run a real prediction.
    Returns a dict with 'fraud_status' and 'fraud_probability'.
    """
    # integrate our model here
    result = predict_transaction(input_data)

    return {
        "fraud_status": "FRAUD DETECTED 🚨" if result['is_fraud'] else "NO FRAUD DETECTED ✅",
        "fraud_probability": result['fraud_probability']
    }
