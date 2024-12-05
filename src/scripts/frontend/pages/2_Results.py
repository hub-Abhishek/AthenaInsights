import streamlit as st
import yaml
import pandas as pd
from misc.utils import load_config, read_df
from misc.frontend_utils import *

# st.logo(st.image("resources/icon.webp"))
st.set_page_config(
    page_title="Athena Insights",
    page_icon="resources/icon.webp",
    layout="wide",)


st.title("Results")

st.button("Click here to refresh results", help='Refresh Results', on_click=refresh_model_results_data, disabled=False)

config = load_config()

models = list_models()

model_selected = st.selectbox(
    "Select the model you're interested in",
    (models),
)

import os
base_dir = 'results/model_data'
directory = f'{base_dir}/{model_selected}'.rstrip('\n')
st.write(directory)
dates = list_directories(directory)

date_selected = st.selectbox(
    "Select the date you're interested in",
    (dates),
)

st.markdown("---")
st.write(f'Displaying resutls for: {model_selected} for the date: {date_selected}')

image_results_dir = f'{directory}/{date_selected}/images'
other_results_dir = f'{directory}/{date_selected}/texts'

y_proba = pd.read_csv(f'{other_results_dir}/y_proba_full.csv').drop(columns='Unnamed: 0').rename(columns={'0': 'A', '1': 'B', '2': 'C'})
dependent_var = pd.read_parquet(f'{base_dir}/dependent_var.parquet').reset_index()
dependent_var = dependent_var[dependent_var.symbol=='SPY']
dependent_var = dependent_var[dependent_var.us_eastern_timestamp.dt.date>=pd.to_datetime(date_selected).date()]

images_generated_during_training = list_files(image_results_dir)
num_images = len(images_generated_during_training)
columns = st.columns(num_images)


if num_images <= 3:
    display_images_in_rows(images_generated_during_training, len(images_generated_during_training), st, image_results_dir)
else:
    display_images_in_rows(images_generated_during_training, 3, st, image_results_dir)

st.markdown("---")

# List of file names to process
file_names = [
    'classification_report_1_day.csv',
    'classification_report_10_day.csv',
    'classification_report_full.csv'
]

# Create columns for the Streamlit app
columns = st.columns(len(file_names))

# Loop through each file, read it, and display it in a column
for column, file_name in zip(columns, file_names):
    file_path = f'{other_results_dir}/{file_name}'
    df = pd.read_csv(file_path).rename(columns={'Unnamed: 0': 'Metric', '0': 'A', '1': 'B', '2': 'C'})
    with column:
        st.write(file_name.replace('_',' ').replace('.csv', ''))
        st.write(df)

st.markdown("---")

# List of file names to process
file_names = [
    'confusion_matrix_rep_1_day.csv',
    'confusion_matrix_rep_10_day.csv',
    'confusion_matrix_rep_full.csv'
]

# Create columns for the Streamlit app
columns = st.columns(len(file_names))

# Loop through each file, read it, and display it in a column
for column, file_name in zip(columns, file_names):
    file_path = f'{other_results_dir}/{file_name}'
    df = pd.read_csv(file_path).rename(columns={'Unnamed: 0': 'Category', '0': 'A', '1': 'B', '2': 'C'})
    with column:
        st.write(file_name.replace('_',' ').replace('.csv', '').replace('rep', 'report'))
        st.write(df)

st.markdown("---")

columns = st.columns(3)

with columns[0]:
    st.write(dependent_var.head())
    st.write(dependent_var.shape)

with columns[1]:
    st.write(y_proba.head())
    st.write(y_proba.shape)