import streamlit as st
import sys
import os
import logging
from app.streamlit.utils import detect_fraud_manual_input

# Ensure the project root is on sys.path so "app" can be imported as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_data_manually():
    with st.sidebar:
        st.header("Input Transaction Data")
        trans_date_time = st.text_input("Transaction Date Time (YYYY-MM-DD HH:MM:SS)", "2019-01-01 00:00:00")
        cc_num = st.text_input("Credit Card Number", "1234567890123456")
        merchant = st.text_input("Merchant", "fraud_Kihn_Abernathy_and_Douglas")
        category = st.selectbox("Category", ["shopping_net", "grocery_pos", "entertainment", "misc_net", "misc_pos"])
        amt = st.slider("Amount", min_value=0.0, max_value=10000.0, value=25.0, step=0.01)
        first_name = st.text_input("First Name", "John")
        last_name = st.text_input("Last Name", "Doe")
        gender = st.selectbox("Gender", ["M", "F"])
        street = st.text_input("Street", "123 Main St")
        city = st.text_input("City", "ExampleCity")
        state = st.text_input("State", "CA")
        zip_code = st.text_input("Zip", "12345")
        lat = st.slider("Latitude", min_value=-90.0, max_value=90.0, value=37.7749, step=0.0001)
        long_ = st.slider("Longitude", min_value=-180.0, max_value=180.0, value=-122.4194, step=0.0001)
        city_pop = st.number_input("City Population", value=100000, step=1)
        job = st.text_input("Job", "Data Scientist")
        dob = st.text_input("DOB (YYYY-MM-DD)", "1990-01-01")
        trans_num = st.text_input("Transaction Number", "abc123xyz")
        unix_time = st.number_input("Unix Time", value=1546300800, step=1)
        merch_lat = st.slider("Merchant Latitude", min_value=-90.0, max_value=90.0, value=37.7749, step=0.0001)
        merch_long = st.slider("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-122.4194, step=0.0001)
    
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

    st.button("RUN" , icon="🤖", on_click=run_btn_handler, args=[input_data])

def main():
   # Main page: Header first
    st.markdown("# Fraud Detection", help="""
        This section allows you to manually input a transaction record and predict its fraud status.
        """)
    
    # Store prediction result in session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Display prediction result if available
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.markdown("### Prediction Result")
        st.markdown(f"**Fraud Status:** {result['fraud_status']}")
        st.markdown(f"**Fraud Probability:** {result['fraud_probability']}")
    
    # Add input form in sidebar
    add_data_manually()

def run_btn_handler(input_data):
    # Store result in session state instead of displaying directly
    st.session_state.prediction_result = detect_fraud_manual_input(input_data)

main()
