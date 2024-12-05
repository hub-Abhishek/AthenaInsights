# Importing necessary libraries
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from misc.utils import load_config, read_df
from misc.frontend_utils import *

# Setting display format for float data
pd.set_option('display.float_format', '{:.10f}'.format)

# Setting up the page configuration
st.set_page_config(
    page_title="Athena Insights",
    page_icon="resources/icon.webp",
    layout="wide",
)

# Display the title of the page
st.title("Results")

# A button to refresh results; currently disabled
st.button("Click here to refresh results", help='Refresh Results', on_click=refresh_model_results_data, disabled=False)

# Load configuration settings
config = load_config()

# Load models list
models = list_models()

# Dropdown to select a model
model_selected = st.selectbox(
    "Select the model you're interested in",
    (models),
)

# Define the base directory for model data
base_dir = 'results/model_data'
directory = f'{base_dir}/{model_selected}'.rstrip('\n')

# Display selected directory path
st.write(directory)

# List available directories for the selected model
dates = list_directories(directory)

# Dropdown to select a date
date_selected = st.selectbox(
    "Select the date you're interested in",
    (dates),
)

# Displaying results section divider
st.markdown("---")

# Define directory paths for images and text results
image_results_dir = f'{directory}/{date_selected}/images'
other_results_dir = f'{directory}/{date_selected}/texts'

# Load probabilities and dependent variables
y_proba = pd.read_csv(f'{other_results_dir}/y_proba_full.csv', dtype={'0': 'float64', '1': 'float64', '2': 'float64'}).drop(columns='Unnamed: 0').rename(columns={'0': 'A', '1': 'B', '2': 'C'})
dependent_var = pd.read_parquet(f'{base_dir}/dependent_var.parquet').reset_index()
dependent_var = dependent_var[dependent_var.us_eastern_timestamp.dt.date>=pd.to_datetime(date_selected).date()].reset_index(drop=True)
results = pd.concat([dependent_var, y_proba], axis=1)
next_day_results = results[results.us_eastern_timestamp.dt.date==pd.to_datetime(date_selected).date()]

# Display images
images_generated_during_training = list_files(image_results_dir)
num_images = len(images_generated_during_training)
if num_images <= 3:
    display_images_in_rows(images_generated_during_training, num_images, st, image_results_dir)
else:
    display_images_in_rows(images_generated_during_training, 3, st, image_results_dir)

# Display section divider
st.markdown("---")

# Loop to display classification reports
file_names = ['classification_report_1_day.csv', 'classification_report_10_day.csv', 'classification_report_full.csv']
columns = st.columns(len(file_names))
for column, file_name in zip(columns, file_names):
    file_path = f'{other_results_dir}/{file_name}'
    df = pd.read_csv(file_path).rename(columns={'Unnamed: 0': 'Metric', '0': 'A', '1': 'B', '2': 'C'})
    with column:
        st.write(file_name.replace('_', ' ').replace('.csv', ''))
        st.write(df)

# Display confusion matrices
file_names = ['confusion_matrix_rep_1_day.csv', 'confusion_matrix_rep_10_day.csv', 'confusion_matrix_rep_full.csv']
columns = st.columns(len(file_names))
for column, file_name in zip(columns, file_names):
    file_path = f'{other_results_dir}/{file_name}'
    df = pd.read_csv(file_path).rename(columns={'Unnamed: 0': 'Category', '0': 'A', '1': 'B', '2': 'C'})
    with column:
        st.write(file_name.replace('_', ' ').replace('rep', 'report').replace('.csv', ''))
        st.write(df)

# Display section divider
st.markdown("---")

# Display plots
st.write('Plots')
st.write(results.tail())

# Call the plot function
plot_categorization(next_day_results, date_selected, st)
