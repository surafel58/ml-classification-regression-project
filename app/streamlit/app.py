import streamlit as st

pg = st.navigation([st.Page("dashboard.py", title="Dashboard", icon="📊"), st.Page("detect.py", title="Fraud Detection", icon="🔍")])
st.set_page_config(page_title="Fraud Detection System", page_icon=":credit_card:", layout="wide")
pg.run()