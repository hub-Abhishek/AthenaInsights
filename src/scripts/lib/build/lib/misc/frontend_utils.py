import boto3
import os
from PIL import Image
from misc.utils import read_df, load_config

def download_dir(prefix, local, bucket_name, client):
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            file_path = obj['Key']
            local_file_path = os.path.join(local, os.path.relpath(file_path, prefix))
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
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
    # path = f"s3://{paths_config['s3_bucket']}/{paths_config['base_folder']}/{paths_config['data_folder']}/{paths_config['data_prep_folder']}/stock_bars_1min.parquet"
    path = 's3://sisyphus-general-bucket/AthenaInsights/latest_data/dependent_variable/stock_bars_1min_base_avg.parquet'
    df = read_df(path)
    df.to_parquet('results/model_data/dependent_var.parquet')
    print('downloaded dependent_var data')

def refresh_model_results_data():

    session = boto3.Session()
    s3 = session.client('s3')
    bucket_name = 'sisyphus-general-bucket'
    prefix = 'AthenaInsights/latest_data/model/results/'
    local_directory = 'results'

    subfolder_names = list_subfolders(bucket_name, prefix, s3)
    results_file_path = 'results/models.txt'
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    with open(results_file_path, 'w') as file:
        for folder_name in sorted(subfolder_names):
            file.write(f"{folder_name}\n")
    print('Written models to file')

    local_directory = 'results/model_data/'
    download_dir(prefix, local_directory, bucket_name, s3)

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