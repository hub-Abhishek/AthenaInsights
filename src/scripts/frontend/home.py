import streamlit as st
import yaml
from misc.utils import load_config

st.set_page_config(
    page_title="Athena Insights",
    page_icon="resources/icon.webp",
    layout="wide",)

st.title("Welcome to Athena Insights!")

st.write("SPY has _N_ fluctuations on a given day. Monitoring these fluctuations can be challenging, and risky. Athena Insights uses ML and AI to monitor these fluctuations for you. Add your number below to recieve sms suggestions on a regular basis, and start your journey into day trading today!")

st.text_input("Contact me at", max_chars=10)
