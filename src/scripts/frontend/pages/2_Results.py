# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from misc.utils import load_config, read_df
from misc.frontend_utils import *
import datetime

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

columns = st.columns(3)

with columns[0]:
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

with columns[1]:
    # List available directories for the selected model
    dates = list_directories(directory)

    # Dropdown to select a date
    date_selected = st.date_input("Select the date you're interested in", 
                                  min_value=pd.to_datetime(dates[0]), 
                                  max_value=pd.to_datetime(dates[-1]), 
                                  value=pd.to_datetime(dates[-1]))

with columns[2]:
    slider = st.slider('Confidence level', min_value=0.0, max_value=1.0, value=0.5)

# Displaying results section divider
st.markdown("---")

# Define directory paths for images and text results
image_results_dir = f'{directory}/{date_selected}/images'
other_results_dir = f'{directory}/{date_selected}/texts'

# Load probabilities and dependent variables
files = list_files(other_results_dir)
dfs = {}
if 'y_proba_1_day.csv' in files:
    dfs['y_proba_1_day'] = pd.read_csv(f'{other_results_dir}/y_proba_1_day.csv', dtype={'0': 'float64', '1': 'float64', '2': 'float64'}).rename(columns={'0': 'A', '1': 'B', '2': 'C'})
elif 'y_proba_10_day.csv' in files:
    dfs['y_proba_10_day'] = pd.read_csv(f'{other_results_dir}/y_proba_10_day.csv', dtype={'0': 'float64', '1': 'float64', '2': 'float64'}).rename(columns={'0': 'A', '1': 'B', '2': 'C'})
elif 'y_proba_full.csv' in files:
    dfs['y_proba_full'] = pd.read_csv(f'{other_results_dir}/y_proba_full.csv', dtype={'0': 'float64', '1': 'float64', '2': 'float64'}).rename(columns={'0': 'A', '1': 'B', '2': 'C'})

# dfs['dependent_var'] = pd.read_parquet(f'{directory}/dependent_var.parquet').reset_index()
# dependent_var = dependent_var[dependent_var.us_eastern_timestamp.dt.date>=pd.to_datetime(date_selected).date()].reset_index(drop=True)
# results = pd.concat([dependent_var, y_proba], axis=1)
# next_day_results = results[results.us_eastern_timestamp.dt.date==pd.to_datetime(date_selected).date()]

# Display images
with st.expander("Model performance plots"):
    images_generated_during_training = list_files(image_results_dir)
    num_images = len(images_generated_during_training)
    if num_images <= 3:
        display_images_in_rows(images_generated_during_training, num_images, st, image_results_dir)
    else:
        display_images_in_rows(images_generated_during_training, 3, st, image_results_dir)

##################################################################################################

# Display plots
with st.expander("Predictions vs reality"):
    if 'y_proba_1_day' in dfs.keys():
        proba_df = dfs['y_proba_1_day'] if 'y_proba_1_day' in dfs.keys() else dfs['y_proba_10_day'] if 'y_proba_10_day' in dfs.keys() else dfs['y_proba_full']
        proba_df['pred_category'] = proba_df.pred.map({0:'A', 1: 'B', 2: 'C'})
        proba_df['category'] = proba_df.actual.map({0:'A', 1: 'B', 2: 'C'})
        proba_df['pred_category_after_confidence'] = (proba_df[['A', 'B', 'C']]>slider).apply(find_true_column, axis=1)
        st.write(proba_df.shape)
        # plot_categorization(proba_df, date_selected, 'close', 'pred_category', st)
        plot_categorization_only_predicted(proba_df, date_selected, 'close', 'pred_category', st)
    else:
        st.write('No data available for the selected date. Please choose a different date.')
    # plot_categorization(y_proba_1_day, date_selected, 'close', 'pred_category_after_confidence', st)

    # st.write(proba_df.head())
    # df = dfs['dependent_var']
    # df = df[df.us_eastern_timestamp.isin(proba_df.us_eastern_timestamp)]
    # df = df[df.us_eastern_timestamp.dt.date==pd.to_datetime(date_selected).date()]
    # df = df[df.market_open]
    # st.write(df.head())
    # st.write(y_proba_1_day.shape, dependent_var[(dependent_var.us_eastern_timestamp.dt.date==pd.to_datetime(date_selected).date())&(dependent_var.mar)].shape)
    
    # st.write(next_day_results[next_day_results.pred_category!='C'])

# Display classification reports, confusion matrices
# with st.expander("Confuion Matrices and other metrics"):
    # display_clasification_report(st, )
    # st.markdown("---")
    # confusion_matrix_rep_1_day(st, other_results_dir)