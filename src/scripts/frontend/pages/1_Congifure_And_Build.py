import streamlit as st
import yaml
from misc.utils import load_config

# st.logo(st.image("resources/icon.webp"))
st.set_page_config(
    page_title="Athena Insights",
    page_icon="resources/icon.webp",
    layout="wide",)

st.title("Configuration Manager")

# def save_config(config, section, data):
#     config[section] = data
#     with open('config.yaml', 'w') as file:
#         yaml.safe_dump(config, file, default_flow_style=False)
#     st.success(f"Changes to {section} saved!")


config = load_config()

# Tabs for each major section of the configuration
tab_keys = ["Common", "Data Ingestion", "Data Prep", "Feature Prep", "Model", "Modeling"]
tabs = st.tabs(tab_keys)

with tabs[0]:
    st.subheader("Common Configuration")
    edited_common = st.text_area("Edit Common Configurations", yaml.safe_dump(config['technical_yaml']['common'], default_flow_style=False), height=250)
    if st.button('Save Common Changes', key='common'):
        new_data = yaml.safe_load(edited_common)
        save_config(config, 'common', new_data)

with tabs[1]:
    st.subheader("Data Ingestion Configuration")
    edited_data_ingestion = st.text_area("Edit Data Ingestion Configurations", yaml.safe_dump(config['technical_yaml']['data_ingestion'], default_flow_style=False), height=250)
    if st.button('Save Data Ingestion Changes', key='data_ingestion'):
        new_data = yaml.safe_load(edited_data_ingestion)
        save_config(config, 'data_ingestion', new_data)

with tabs[2]:
    st.subheader("Data Preparation Configuration")
    edited_data_prep = st.text_area("Edit Data Prep Configurations", yaml.safe_dump(config['technical_yaml']['data_prep'], default_flow_style=False), height=300)
    if st.button('Save Data Prep Changes', key='data_prep'):
        new_data = yaml.safe_load(edited_data_prep)
        save_config(config, 'data_prep', new_data)

with tabs[3]:
    st.subheader("Feature Preparation Configuration")
    edited_feature_prep = st.text_area("Edit Feature Prep Configurations", yaml.safe_dump(config['technical_yaml']['feature_prep'], default_flow_style=False), height=400)
    if st.button('Save Feature Prep Changes', key='feature_prep'):
        new_data = yaml.safe_load(edited_feature_prep)
        save_config(config, 'feature_prep', new_data)

with tabs[4]:
    st.subheader("Model Configuration")
    edited_model = st.text_area("Edit Model Configurations", yaml.safe_dump(config['technical_yaml']['model'], default_flow_style=False), height=150)
    if st.button('Save Model Changes', key='model'):
        new_data = yaml.safe_load(edited_model)
        save_config(config, 'model', new_data)

with tabs[5]:
    st.subheader("Modeling Configuration")
    edited_modeling = st.text_area("Edit Modeling Configurations", yaml.safe_dump(config['technical_yaml']['modeling'], default_flow_style=False), height=250)
    if st.button('Save Modeling Changes', key='modeling'):
        new_data = yaml.safe_load(edited_modeling)
        save_config(config, 'modeling', new_data)