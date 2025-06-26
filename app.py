import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try different import methods
try:
    from utils import fetch_all_funds, fetch_fund_data, calculate_metrics, benchmark_metrics
    st.success("✅ Successfully imported utility functions")
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please make sure utils.py is in the same directory as app.py")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.stop()

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# Load all available funds once
@st.cache_data
def load_all_funds():
    return fetch_all_funds()

try:
    all_funds = load_all_funds()
    if not all_funds:
        st.error("❌ Unable to fetch fund list. Please check your internet connection.")
        st.stop()
    
    fund_options = {f"{fund['schemeName']} ({fund['schemeCode']})": fund['schemeCode'] 
                   for fund in all_funds}
    
    st.success(f"✅ Loaded {len(all_funds)} mutual funds")
except Exception as e:
    st.error(f"❌ Error loading funds: {str(e)}")
    st.stop()

# Rest of your app logic here...
# (Copy the rest from the standalone version