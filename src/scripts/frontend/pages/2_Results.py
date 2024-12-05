import streamlit as st
import yaml
from misc.utils import load_config

st.logo(st.image("resources/icon.webp"))
st.set_page_config(
    page_title="Athena Insights",
    page_icon="resources/icon.webp",
    layout="wide",)

config = load_config()
