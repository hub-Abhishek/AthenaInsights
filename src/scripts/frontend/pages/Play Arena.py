# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from misc.utils import load_config, read_df
from misc.frontend_utils import *
from misc.features_calc import *
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

        
# Load configuration settings
config = load_config()

df = pd.read_parquet(f"results/model_data/{config['technical_yaml']['common']['model_name']}/dependent_var.parquet")
df = df.reset_index()
dates = df.us_eastern_timestamp.dt.date.unique()
min_date = min(dates)
max_date = max(dates)

col1, col2 = st.columns(2)

with col1:
    # A button to refresh results; currently disabled
    st.button("Click here to refresh target file", help='Refresh Results', on_click=refresh_target, disabled=False)
    
with col2:
    st.text('Select a date')  # Display the key        
    st.write(f'dates available from {min_date} to {max_date}')
    date_selected = st.date_input("Select the date you're interested in", 
                                min_value=min_date, 
                                max_value=max_date, )
    if date_selected not in dates:
        raise ValueError("selected date not available")
    
with st.expander("Feature Preparation Configurations"):
    
    col1, col2, col3 = st.columns(3)
    key_1 = 'dependent_var'
    key_2 = 'based_on'
    with col1:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        based_on = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        st.write('---')
        
    key_2 = 'prev_data_points'
    with col2:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        prev_data_points = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        prev_data_points = int(prev_data_points)
        st.write('---')
        
    key_2 = 'positive_slope_threshold'
    with col3:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        positive_slope_threshold = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        positive_slope_threshold = float(positive_slope_threshold)
        st.write('---')
        
    key_2 = 'negative_slope_threshold'
    with col1:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        negative_slope_threshold = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        negative_slope_threshold = float(negative_slope_threshold)
        st.write('---')
        
    key_2 = 'positive_rise_threshold'
    with col2:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        positive_rise_threshold = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        positive_rise_threshold = float(positive_rise_threshold)
        st.write('---')
        
    key_2 = 'negative_drop_threshold'
    with col3:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        negative_drop_threshold = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        negative_drop_threshold = float(negative_drop_threshold)
        st.write('---')
        
    key_2 = 'positive_future_window'
    with col1:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        positive_future_window = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        positive_future_window = int(positive_future_window)
        st.write('---')
        
    key_2 = 'negative_future_window'
    with col2:
        st.text(key_1 + ': ' + key_2)  # Display the key
        st.text(config['technical_yaml']['feature_prep'][key_1][key_2])  # Display the key
        negative_future_window = st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        negative_future_window = int(negative_future_window)
        st.write("---")

    # with col3:
    #     st.text('Select a date')  # Display the key        
    #     st.write(f'dates available from {min_date} to {max_date}')
    #     date_selected = st.date_input("Select the date you're interested in", 
    #                                 min_value=min_date, 
    #                                 max_value=max_date, )
    #     if date_selected not in dates:
    #         raise ValueError("selected date not available")
        
    # key_2 = 'se'

    #     dates = list_directories(directory)
    #     date_series = pd.date_range(start='2024-01-01', end='end_date', freq='D')
    #     date_series = [z.strftime('%Y-%m-%d') for z in date_series]

    # # Dropdown to select a date
    # date_selected = st.date_input("Select the date you're interested in", 
    #                               min_value=pd.to_datetime(dates[0]), 
    #                               max_value=pd.to_datetime(dates[-1]), 
    #                               value=pd.to_datetime(dates[-1]))
    #     st.text_input(f"Updated {key_1}: {key_2}", value=config['technical_yaml']['feature_prep'][key_1][key_2])
        # st.write("---")

###############################################
###############################################
###############################################

selected_df = df[df.us_eastern_timestamp.dt.date==date_selected]
selected_df = selected_df[selected_df.market_open]
prev_day_close = df[(df.us_eastern_timestamp.dt.date<date_selected)&(df.market_open)].close.iloc[-1]

categories, future_highs, future_lows, slopes = categorize_points(selected_df,
                                                                  field=based_on, 
                                                                  prev_data_points=prev_data_points, 
                                                                  positive_slope_threshold=positive_slope_threshold, 
                                                                  negative_slope_threshold=negative_slope_threshold, 
                                                                  positive_rise_threshold=positive_rise_threshold, 
                                                                  negative_drop_threshold=negative_drop_threshold, 
                                                                  positive_future_window=positive_future_window, 
                                                                  negative_future_window=negative_future_window)

selected_df['category'] = categories
selected_df['future_highs'] = future_highs
selected_df['future_lows'] = future_lows
selected_df['slopes'] = slopes

increase_05 = prev_day_close * 1.005
decrease_05 = prev_day_close * 0.995
increase_1 = prev_day_close * 1.01
decrease_1 = prev_day_close * 0.99

# ## first image
fig, axs = plt.subplots(1, 1, figsize=(14, 7))
x = np.asarray(selected_df.us_eastern_timestamp, dtype='datetime64[s]')
axs.plot(x, selected_df['close_sma_5m'], label='close price', color='gray', linewidth=2)
axs.axhline(y=prev_day_close, color='black', linewidth=1, label='yest close') # 'linestyle='--'
axs.axhline(y=increase_05, color='black', linewidth=.5, linestyle='--', label='+0.5%')
axs.axhline(y=decrease_05, color='black', linewidth=.5, linestyle='--', label='-0.5%')
axs.axhline(y=increase_1, color='black', linewidth=.5, linestyle='--', label='+1%')
axs.axhline(y=decrease_1, color='black', linewidth=.5, linestyle='--', label='-1%')
for cat, color in zip(['A', 'B', 'C'], ['green', 'red', 'gray']):
    axs.scatter(selected_df[selected_df['category'] == cat].us_eastern_timestamp,
                    selected_df[selected_df['category'] == cat]['close_sma_5m'],
                    color=color, label=f'Category {cat}',
                    s=30 if cat != 'C' else 0)
axs.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]))
axs.tick_params(axis='x', labelrotation=90)
axs.grid(True) 


# plt.legend()
plt.title(f'Price Categorization on {date_selected}')
plt.xlabel('Timestamp')
# plt.ylabel(f'Close Price')
st.pyplot(plt)

# ## second image
fig, axs = plt.subplots(1, 1, figsize=(14, 7))
axs.plot(x, selected_df['close_sma_5m'], label='close price', color='black', linewidth=1)
axs.axhline(y=prev_day_close, color='black', linewidth=1, label='yest close') # 'linestyle='--'
axs.axhline(y=increase_05, color='black', linewidth=1, linestyle='--', label='+0.5%')
axs.axhline(y=decrease_05, color='black', linewidth=1, linestyle='--', label='-0.5%')
axs.axhline(y=increase_1, color='black', linewidth=1, linestyle='--', label='+1%')
axs.axhline(y=decrease_1, color='black', linewidth=1, linestyle='--', label='-1%')
axs.plot(x, selected_df['future_highs'], label=f'furute highs', color='green', linewidth=0.5)
axs.plot(x, selected_df['future_lows'], label=f'furute lows', color='red', linewidth=0.5)
for cat, color in zip(['A', 'B', 'C'], ['green', 'red', 'gray']):
    axs.scatter(selected_df[selected_df['category'] == cat].us_eastern_timestamp,
                    selected_df[selected_df['category'] == cat]['close_sma_5m'],
                    color=color, label=f'Category {cat}',
                    s=30 if cat != 'C' else 0)
# axs.plot(x, selected_df['slopes'], label=f'slopes', color='blue', linewidth=1)

axs_secondary_axs = axs.twinx()
axs_secondary_axs.plot(x, selected_df['slopes'], color='blue', linewidth=1, linestyle='-.')
axs_secondary_axs.axhline(y=positive_slope_threshold, color='red', linewidth=.5, linestyle='-.', label='positive_slope_threshold')
axs_secondary_axs.text(x.max(), positive_slope_threshold, f'{positive_slope_threshold}', verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10)
axs_secondary_axs.axhline(y=negative_slope_threshold, color='red', linewidth=.5, linestyle='-.', label='negative_slope_threshold')
axs_secondary_axs.text(x.max(), negative_slope_threshold, f'{negative_slope_threshold}', verticalalignment='top', horizontalalignment='right', color='blue', fontsize=10)
axs_secondary_axs.set_ylim(-0.1, 0.1)
axs_secondary_axs.tick_params(axis='y', length=0)
# axs_secondary_axs.set_ylabel('Slope', color='blue')
axs.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
axs_secondary_axs.set_yticklabels([])

axs.tick_params(axis='x', labelrotation=90)
axs.grid(True) 

# plt.legend()
plt.title(f'Price Categorization on {date_selected}')
plt.xlabel('Timestamp')
# plt.ylabel(f'Close Price')
st.pyplot(plt)


# fig, axs = plt.subplots(1, 1, figsize=(14, 7))
# axs.plot(x, selected_df['slopes'], color='blue', linewidth=1, linestyle='-.')
# axs.axhline(y=positive_slope_threshold, color='blue', linewidth=.5, linestyle='-.', label='positive_slope_threshold')
# # axs.text(x.max(), positive_slope_threshold, f'{positive_slope_threshold}', verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10)
# axs.axhline(y=negative_slope_threshold, color='red', linewidth=.5, linestyle='-.', label='negative_slope_threshold')
# # axs.text(x.max(), negative_slope_threshold, f'{negative_slope_threshold}', verticalalignment='top', horizontalalignment='right', color='blue', fontsize=10)
# st.pyplot(plt)

st.write(selected_df)