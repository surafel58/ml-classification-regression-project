import streamlit as st

# Set page config as the VERY FIRST Streamlit command
st.set_page_config(
    page_title="Fraud Detection System", 
    page_icon=":credit_card:", 
    layout="wide"
)

# Import other modules AFTER setting page config
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create navigation AFTER imports
pg = st.navigation([
    st.Page("dashboard.py", title="Dashboard", icon="📊"), 
    st.Page("detect.py", title="Fraud Detection", icon="🔍")
])
pg.run()