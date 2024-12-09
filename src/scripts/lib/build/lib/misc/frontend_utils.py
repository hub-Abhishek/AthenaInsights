import boto3
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from misc.utils import read_df, load_config, log

def download_dir(prefix, local, bucket_name, client):
    paginator = client.get_paginator('list_objects_v2')
    log(f'prefix - {prefix}')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            file_path = obj['Key']
            local_file_path = os.path.join(local, os.path.relpath(file_path, prefix))
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            log(f'downloading {file_path} to {local_file_path}')
            client.download_file(bucket_name, file_path, local_file_path)

def list_subfolders(bucket_name, prefix, client):
    subfolders = set()
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        for prefix_info in page.get('CommonPrefixes', []):  # Filter the prefixes (subfolders)
            subfolder_path = prefix_info['Prefix']
            subfolder_name = subfolder_path.strip('/').split('/')[-1]
            subfolders.add(subfolder_name)
    return subfolders

def refresh_target():
    config = load_config()
    paths_config = config['paths_config']
    path = f"s3://{paths_config['s3_bucket']}/{paths_config['base_folder']}/{paths_config['data_folder']}/{config['technical_yaml']['common']['model_name']}/dependent_variable/stock_bars_1min_base_avg.parquet"
    df = read_df(path)
    df.to_parquet(f'results/model_data/{config["technical_yaml"]["common"]["model_name"]}/dependent_var.parquet')
    log('downloaded dependent_var data')

def refresh_model_results_data():

    bucket_name = 'sisyphus-general-bucket'
    prefix = 'AthenaInsights/latest_data'
    local_directory = 'results'
    results_file_path = 'results/models.txt'
    forbidden_subdirectories = ['csv', 'parquet']

    session = boto3.Session()
    s3 = session.client('s3')
    subfolder_names = list_subfolders(bucket_name, prefix + '/', s3)
    subfolder_names = subfolder_names - set(forbidden_subdirectories)
    log(f'subfolder_names detected - {subfolder_names}')
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    with open(results_file_path, 'w') as file:
        for folder_name in sorted(subfolder_names):
            file.write(f"{folder_name}\n")
    log('Written models to file')

    local_directory = 'results/model_data/'
    for subfolder in subfolder_names:
        log(f'downloading {subfolder}')
        download_dir(f'{prefix}/{subfolder}/model/results/', f'{local_directory}/{subfolder}', bucket_name, s3)

    refresh_target()

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def list_models():
    file_path = 'results/models.txt'
    model_list = read_txt_file(file_path)
    return model_list

def list_directories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def display_images_in_rows(images_generated_during_training, images_per_row, st, image_folder):
    for i in range(0, len(images_generated_during_training), images_per_row):
        row_images = images_generated_during_training[i:i+images_per_row]
        columns = st.columns(len(row_images))
        for idx, image_file in enumerate(row_images):
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            with columns[idx]:
                st.image(image, caption=image_file, use_column_width=True)

# Plot categorization results
def plot_categorization(df, date_selected, dependent_field_name, predicted_field_name, st):
    """Plot categorization for a given day with dynamic field selection."""
    plt.figure(figsize=(14, 7))
    fig, axs = plt.subplots(2, 1, figsize=(14, 14))
    axs[0].plot(df.us_eastern_timestamp, df[dependent_field_name], label=f'Close Price', color='gray', linewidth=2)
    for cat, color in zip(['A', 'B', 'C'], ['green', 'red', 'gray']):
        axs[0].scatter(df[df['category'] == cat].us_eastern_timestamp,
                       df[df['category'] == cat][dependent_field_name],
                       color=color, label=f'Category {cat}',
                       s=30 if cat != 'C' else 0)
    axs[0].grid(axis='x', which='major', linestyle=':', linewidth='0.5', color='gray')
    axs[0].grid(axis='x', which='minor', linestyle=':', linewidth='0.5', color='gray')
    # axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=10))

    axs[1].plot(df.us_eastern_timestamp, df[dependent_field_name], label=f'Close Price', color='gray', linewidth=2)
    for cat, color in zip(['A', 'B', 'C'], ['green', 'red', 'gray']):
        axs[1].scatter(df[df[predicted_field_name] == cat].us_eastern_timestamp,
                       df[df[predicted_field_name] == cat][dependent_field_name],
                       color=color, label=f'Preds {cat}',
                       s=20 if cat != 'C' else 0)
    axs[1].grid(axis='x', which='major', linestyle=':', linewidth='0.5', color='gray')
    axs[1].grid(axis='x', which='minor', linestyle=':', linewidth='0.5', color='gray')
    # axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=10))

    plt.legend()
    plt.title(f'Price Categorization on {date_selected}')
    plt.xlabel('Timestamp')
    plt.ylabel(f'Close Price')
    st.pyplot(plt)

def find_true_column(row):
    for col in ['A', 'B', 'C']:
        if row[col]:
            return col
    return 'C'

def display_clasification_report(st, other_results_dir):
    file_names = ['classification_report_1_day.csv', 'classification_report_10_day.csv', 'classification_report_full.csv']
    columns = st.columns(len(file_names))
    for column, file_name in zip(columns, file_names):
        file_path = f'{other_results_dir}/{file_name}'
        df = pd.read_csv(file_path).rename(columns={'Unnamed: 0': 'Metric', '0': 'A', '1': 'B', '2': 'C'})
        with column:
            st.write(file_name.replace('_', ' ').replace('.csv', ''))
            st.write(df)

def confusion_matrix_rep_1_day(st, other_results_dir):
    file_names = ['confusion_matrix_rep_1_day.csv', 'confusion_matrix_rep_10_day.csv', 'confusion_matrix_rep_full.csv']
    columns = st.columns(len(file_names))
    for column, file_name in zip(columns, file_names):
        file_path = f'{other_results_dir}/{file_name}'
        df = pd.read_csv(file_path).rename(columns={'Unnamed: 0': 'Category', '0': 'A', '1': 'B', '2': 'C'})
        with column:
            st.write(file_name.replace('_', ' ').replace('rep', 'report').replace('.csv', ''))
            st.write(df)

def save_technical_yaml(config, key):
    technical_yaml_filepath = f'config/{config["technical_yaml"]["common"]["model_name"]}/technical.yaml'
    with open(technical_yaml_filepath, 'w') as file:
        yaml.safe_dump(data, file)