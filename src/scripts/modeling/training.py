import os
import re
import boto3
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report

from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, read_df, save_df_as_parquet, save_df_as_csv, get_all_paths_from_loc, get_name_and_type
from misc.features_calc import update_config, load_features_config, get_paths_and_cols_from_config, check_index_is_monotonic_increasing
# from misc.features_calc import 
warnings.filterwarnings("ignore")


class ModelTraining(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'ModelTrainingFilePrep'
        log(f'running {self.name}')
        if config is None:
            self.config = load_config()
        else:
            self.config = config

        self.paths_config = self.config['paths_config']

        self.bucket_loc = f's3://{self.paths_config["s3_bucket"]}'
        self.base_folder = self.paths_config["base_folder"]
        self.data_folder = self.paths_config["data_folder"]
        self.data_prep_folder = self.paths_config["data_prep_folder"]
        self.reduced_autocorelation_folder = self.paths_config["reduced_autocorelation_folder"]
        self.feature_prep_folder = self.paths_config["feature_prep_folder"]

        self.model_base_folder = self.paths_config["model_base_folder"]
        self.model_data_folder = self.paths_config["model_data_folder"]
        self.model_results_folder = self.paths_config["model_results_folder"]
        self.model_results_images_subfolder = self.paths_config["model_results_images_subfolder"]
        self.model_results_texts_subfolder = self.paths_config["model_results_texts_subfolder"]

        self.client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self.bucket_resource = self.s3_resource.Bucket(self.paths_config["s3_bucket"])

        self.common_config = self.config['technical_yaml']['common']
        self.feature_prep = self.config['technical_yaml']['feature_prep']
        self.modeling_config = self.config['technical_yaml']['modeling']

        self.features_to_be_calculated = self.feature_prep['features_to_be_calculated']
        self.symbols = self.feature_prep['symbols']
        self.calc_metrics = self.feature_prep['calc_metrics']

        self.model_name = self.common_config['model_name']
        self.model_base_path = f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/{self.paths_config["model_base_folder"]}'
        self.model_results_path = f'{self.model_base_path}/{self.model_results_folder}'
        self.training_file_path = f'{self.model_base_path}/{self.paths_config["model_data_folder"]}'

        self.features_listing_config = load_features_config()

    def extract(self):

        model = xgb.XGBClassifier(n_estimators=100,
                                objective='multi:softmax',
                                n_jobs =-1,
                                random_state=420,
                                num_class=3,
                                eval_metric=['merror','mlogloss'])

        training_files = []
        # log(f'training files config - {self.features_listing_config["training_files"]}')
        
        for k in self.features_listing_config['training_files'].keys():
            for file in self.features_listing_config['training_files'][k].keys():
                training_files.append(file)
        if len(training_files)==1:
            df = read_df(self.features_listing_config['training_files'][k][file]['path'])
        else:
            log(f'training files - {training_files}')
            raise ValueError('which training file? please check features config')


        # category_map = self.modeling_config['category_map']
        # reverse_category_map = {v: k for k, v in category_map.items()}
        # df['mapped_category'] = df['category'].map({'A': 0, 'B': 1, 'C':2})
        # df['mapped_category'].value_counts()
        # df = df.drop(columns=['symbol', 'symbol1', 'category'])
        # df = pd.concat([df.drop(columns='direction'), pd.get_dummies(df['direction'], drop_first=True)], axis=1)
        df = df.fillna(0)
        log(f'train_on_market_open_only - {self.modeling_config["train_on_market_open_only"]}, predict_on_market_open_only - {self.modeling_config["predict_on_market_open_only"]}')
        if self.modeling_config['train_on_market_open_only']:
            df = df['market_open']

        start_date = self.modeling_config['start_date']
        end_date = self.modeling_config['end_date']
        date_series = pd.date_range(start=start_date, end=end_date, freq='D')
        date_series = [z.strftime('%Y-%m-%d') for z in date_series]

        log(f'start_date - {start_date}')
        log(f'end_date - {end_date}')
        log(f'dates - {date_series}')

        return {
            'df': df,
            'model': model,
            'date_series': date_series
        }

    @staticmethod
    def get_dates(dt):
        test_date = dt
        next_day = (datetime.datetime.strptime(test_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        next_10_day = (datetime.datetime.strptime(test_date, '%Y-%m-%d') + datetime.timedelta(days=9)).strftime('%Y-%m-%d')
        prev_day = (datetime.datetime.strptime(test_date, '%Y-%m-%d') + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        return test_date, next_day, next_10_day, prev_day

    @staticmethod
    def get_train_test_split(df, test_date, next_day, next_10_day, prev_day, predict_on_market_open_only):

        df = df.drop(columns=df.select_dtypes(include='object').columns)
        X_train = df.loc[:prev_day, ].drop(columns=['mapped_category'])
        y_train = df.loc[:prev_day, 'mapped_category']

        if predict_on_market_open_only is True:
            df = df[df.market_open]
            
        X_test_only_next_day = df.loc[test_date, ].drop(columns=['mapped_category'])
        y_test_only_next_day = df.loc[test_date, 'mapped_category']
            

        X_test_next_10_days = df.loc[test_date: next_10_day, ].drop(columns=['mapped_category'])
        y_test_next_10_days = df.loc[test_date: next_10_day, 'mapped_category']

        X_test_full = df.loc[test_date:, ].drop(columns=['mapped_category'])
        y_test_full = df.loc[test_date:, 'mapped_category']

        return X_train, y_train, X_test_only_next_day, y_test_only_next_day, X_test_next_10_days, y_test_next_10_days, X_test_full, y_test_full

    @staticmethod
    def initialte_and_train(model, X_train, y_train, X_test_only_next_day, y_test_only_next_day, 
                            X_test_next_10_days, y_test_next_10_days, X_test_full, y_test_full):

        model.fit(X_train,
                y_train,
                verbose=0,
                eval_set=[(X_train, y_train), (X_test_only_next_day, y_test_only_next_day), (X_test_next_10_days, y_test_next_10_days), (X_test_full, y_test_full)])

        return model

    def save_model(self, model, models_subfolder):
        s3 = boto3.client('s3')
        bucket_name = self.paths_config["s3_bucket"]
        model_path = f'{models_subfolder}/my_model.model'
        model_path = model_path.replace(f'{self.bucket_loc}/', '')
        log(f'model_path - {model_path}')

        buffer = BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket_name, model_path)

#         import boto3
#         from io import BytesIO
#         import pickle

#         # Initialize a boto3 client
#         s3 = boto3.client('s3')
#         bucket_name = 'your-bucket-name'
#         model_path = 'models/my_model.pkl'

#         # Download the object from S3 to a buffer
#         buffer = BytesIO()
#         s3.download_fileobj(bucket_name, model_path, buffer)
#         buffer.seek(0)

#         # Load the model from the buffer using pickle
#         loaded_bst = pickle.load(buffer)
#         print('Model loaded successfully')

    def save_image(self, img_data, name):
        name = name.replace(f'{self.bucket_loc}/', '')
        log(f'saving image to - {name}')
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        self.bucket_resource.put_object(Body=img_data, ContentType='image/png', Key=name)

#         import boto3
#         from io import BytesIO
#         import pickle

#         # Initialize a boto3 client
#         s3 = boto3.client('s3')
#         bucket_name = 'your-bucket-name'
#         accuracy_path = 'accuracy/accuracy_score.pkl'

#         # Download the object from S3 to a buffer
#         buffer = BytesIO()
#         s3.download_fileobj(bucket_name, accuracy_path, buffer)
#         buffer.seek(0)

#         # Load the accuracy score from the buffer using pickle
#         loaded_accuracy = pickle.load(buffer)
#         print(f'Loaded Accuracy Score: {loaded_accuracy}')

    def upload_config(self):
        log(f'uploading config files to {self.model_results_path}/paths.yaml')
        log(os.path.exists(f'config/{self.model_name}/paths.yaml'))
        self.client.upload_file(f'config/{self.model_name}/paths.yaml', 
                                self.paths_config['s3_bucket'], 
                                f"{self.model_results_path}/paths.yaml".replace('s3://', '').replace(self.paths_config['s3_bucket'], ''))
        self.client.upload_file(f'config/{self.model_name}/features.yaml', 
                                self.paths_config['s3_bucket'], 
                                f"{self.model_results_path}/features.yaml".replace('s3://', '').replace(self.paths_config['s3_bucket'], ''))
        self.client.upload_file(f'config/{self.model_name}/technical.yaml', 
                                self.paths_config['s3_bucket'], 
                                f"{self.model_results_path}/technical.yaml".replace('s3://', '').replace(self.paths_config['s3_bucket'], ''))

    def save_train_plots(self, clf, dt, images_subfolder):
        results = clf.evals_result()
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)

        # xgboost 'mlogloss' plot
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
        ax.plot(x_axis, results['validation_2']['mlogloss'], label='Test_full')
        ax.plot(x_axis, results['validation_3']['mlogloss'], label='Test_only_next_day')
        ax.legend()
        plt.ylabel('mlogloss')
        plt.title(f'GridSearchCV XGBoost mlogloss - {dt}')
        self.save_image(plt, f'{images_subfolder}/GridSearchCV XGBoost mlogloss - {dt}.png')
        # fig.savefig(f'{self.images_directory}/XGBoost mlogloss - {dt}.png')

        # xgboost 'merror' plot
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(x_axis, results['validation_0']['merror'], label='Train')
        ax.plot(x_axis, results['validation_1']['merror'], label='Test')
        ax.plot(x_axis, results['validation_2']['merror'], label='Test_full')
        ax.plot(x_axis, results['validation_3']['merror'], label='Test_only_next_day')
        ax.legend()
        plt.ylabel('merror')
        plt.title(f'GridSearchCV XGBoost merror - {dt}')
        self.save_image(plt, f'{images_subfolder}/XGBoost merror - {dt}.png')

    def generate_reports(self, x, y, clf, dur, texts_subfolder):
        y_p = clf.predict(x)

        y_proba = clf.predict_proba(x)
        y_proba = pd.DataFrame(y_proba)
        y_proba.index = y.index
        y_proba['pred'] = y_p
        y_proba['actual'] = y.values
        y_proba['market_open'] = x.market_open
        y_proba['close'] = x.close
        if 'close_diff_ema_5m' in x.columns:
            y_proba['close_diff_ema_5m'] = x.close_diff_ema_5m
        elif 'close_ema_5m' in x.columns:
            y_proba['close_ema_5m'] = x.close_ema_5m
        if 'close_diff_sma_5m' in x.columns:
            y_proba['close_diff_sma_5m'] = x.close_diff_sma_5m
        elif 'close_sma_5m' in x.columns:
            y_proba['close_sma_5m'] = x.close_sma_5m
        # import pdb;pdb.set_trace();
        save_df_as_csv(y_proba, f'{texts_subfolder}/y_proba_{dur}.csv')

    def train(self, dfs):
        df = dfs['df']
        model = dfs['model']
        date_series = dfs['date_series']

        for dt in tqdm(date_series):
            log(f"running for dt = {dt}")

            models_subfolder = f"{self.model_results_path}/{dt}"
            images_subfolder = f"{models_subfolder}/{self.model_results_images_subfolder}"
            texts_subfolder = f"{models_subfolder}/{self.model_results_texts_subfolder}"

            test_date, next_day, next_10_day, prev_day = self.get_dates(dt)
            log(f"test_date: {test_date}, next_day: {next_day}, next_10_day: {next_10_day}, prev_day: {prev_day}")

            X_train, y_train, X_test_only_next_day, y_test_only_next_day, X_test_next_10_days, y_test_next_10_days, X_test_full, y_test_full = self.get_train_test_split(df, test_date, next_day, next_10_day, prev_day, self.modeling_config["predict_on_market_open_only"])
            
            if y_test_full.empty:
                break

            log("training the model")
            clf = self. initialte_and_train(model, X_train, y_train, X_test_only_next_day, y_test_only_next_day, X_test_next_10_days, y_test_next_10_days, X_test_full, y_test_full)
            self.save_model(clf, models_subfolder)
            log(f"model trained for {dt}")

            log("results")
            self.save_train_plots(clf, dt, images_subfolder)

            if not X_test_only_next_day.empty:
                log("1 day test")
                self.generate_reports(X_test_only_next_day, y_test_only_next_day, clf, '1_day', texts_subfolder)
                # self.generate_reports(X_test_only_next_day, y_test_only_next_day, clf, '1_day', dt, texts_subfolder)

            if not X_test_next_10_days.empty:
                log("10 day test")
                self.generate_reports(X_test_next_10_days, y_test_next_10_days, clf, '10_day', texts_subfolder)

            if not X_test_full.empty:
                log("full test")
                self.generate_reports(X_test_full, y_test_full, clf, 'full', texts_subfolder)
                
            log("feature importances")
            fea_imp = pd.DataFrame(data=list(clf.feature_importances_ ), index=list(X_train.columns), columns=["score"]).sort_values(by = "score", ascending=True)
            save_df_as_csv(fea_imp, f'{texts_subfolder}/feature_importances.csv')
            
            del clf, X_train, y_train, X_test_only_next_day, y_test_only_next_day, X_test_next_10_days, y_test_next_10_days, X_test_full, y_test_full, fea_imp

        self.upload_config()

    def run(self, ):
        dfs = self.extract()
        self.train(dfs)

if __name__ == '__main__':
    config = load_config()
    model_training = ModelTraining(config)
    model_training.run()
    # model_training.upload_config()
    
    

#     def save_smaller_reports(self, report, name):
#         name = name.replace(f'{self.bucket_loc}/', '')
#         log(f'report path - {name}')
#         s3 = boto3.client('s3')
#         bucket_name = self.paths_config["s3_bucket"]

#         buffer = BytesIO()
#         pickle.dump(report, buffer)
#         buffer.seek(0)
#         s3.upload_fileobj(buffer, bucket_name, name)
            
#     def generate_reports(self, x, y, clf, dur, dt, texts_subfolder, df=None):
#         y_p = clf.predict(x)
        
#         y_proba = clf.predict_proba(x)
#         y_proba = pd.DataFrame(y_proba)
#         y_proba['pred'] = y_p
#         save_df_as_csv(y_proba, f'{texts_subfolder}/y_proba_{dur}.csv')

#         confusion_matrix_rep = pd.DataFrame(confusion_matrix(y, y_p))
#         save_df_as_csv(confusion_matrix_rep, f'{texts_subfolder}/confusion_matrix_rep_{dur}.csv')

#         accuracy_score_rep = accuracy_score(y, y_p)
#         self.save_smaller_reports(accuracy_score_rep, name=f'{texts_subfolder}/accuracy_score_rep_{dur}.pkl')

#         balanced_accuracy_score_rep = balanced_accuracy_score(y, y_p)
#         self.save_smaller_reports(balanced_accuracy_score_rep, name=f'{texts_subfolder}/balanced_accuracy_score_rep_{dur}.pkl')

#         micro_precision_score_rep = precision_score(y, y_p, average='micro')
#         self.save_smaller_reports(micro_precision_score_rep, name=f'{texts_subfolder}/micro_precision_score_rep_{dur}.pkl')

#         micro_recall_score_rep = recall_score(y, y_p, average='micro')
#         self.save_smaller_reports(micro_recall_score_rep, name=f'{texts_subfolder}/micro_recall_score_rep_{dur}.pkl')

#         micro_f1_score_rep = f1_score(y, y_p, average='micro')
#         self.save_smaller_reports(micro_f1_score_rep, name=f'{texts_subfolder}/micro_f1_score_rep_{dur}.pkl')

#         macro_precision_score_rep = precision_score(y, y_p, average='macro')
#         self.save_smaller_reports(macro_precision_score_rep, name=f'{texts_subfolder}/macro_precision_score_rep_{dur}.pkl')

#         macro_recall_score_rep = recall_score(y, y_p, average='macro')
#         self.save_smaller_reports(macro_recall_score_rep, name=f'{texts_subfolder}/macro_recall_score_rep_{dur}.pkl')

#         macro_f1_score_rep = f1_score(y, y_p, average='macro')
#         self.save_smaller_reports(macro_f1_score_rep, name=f'{texts_subfolder}/macro_f1_score_rep_{dur}.pkl')

#         weighted_precision_score_rep = precision_score(y, y_p, average='weighted')
#         self.save_smaller_reports(weighted_precision_score_rep, name=f'{texts_subfolder}/weighted_precision_score_rep_{dur}.pkl')

#         weighted_recall_score_rep = recall_score(y, y_p, average='weighted')
#         self.save_smaller_reports(weighted_recall_score_rep, name=f'{texts_subfolder}/weighted_recall_score_rep_{dur}.pkl')

#         weighted_f1_score_rep = f1_score(y, y_p, average='weighted')
#         self.save_smaller_reports(weighted_f1_score_rep, name=f'{texts_subfolder}/weighted_f1_score_rep_{dur}.pkl')

#         classification_report_rep = pd.DataFrame(classification_report(y, y_p, output_dict=True))
#         save_df_as_csv(classification_report_rep, f'{texts_subfolder}/classification_report_{dur}.csv')

#         log('\n-------------------- Key Metrics --------------------')
#         log('\nAccuracy: {:.2f}'.format(accuracy_score_rep))
#         log('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score_rep))
#         log('Micro Precision: {:.2f}'.format(micro_precision_score_rep))
#         log('Micro Recall: {:.2f}'.format(micro_recall_score_rep))
#         log('Micro F1-score: {:.2f}\n'.format(micro_f1_score_rep))
#         log('Macro Precision: {:.2f}'.format(macro_precision_score_rep))
#         log('Macro Recall: {:.2f}'.format(macro_recall_score_rep))
#         log('Macro F1-score: {:.2f}\n'.format(macro_f1_score_rep))
#         log('Weighted Precision: {:.2f}'.format(weighted_precision_score_rep))
#         log('Weighted Recall: {:.2f}'.format(weighted_recall_score_rep))
#         log('Weighted F1-score: {:.2f}'.format(weighted_f1_score_rep))
#         log('\n--------------- Classification Report ---------------\n')
#         log(classification_report_rep)




# def log(x, dt, image=None, add=None):
#     if type(x)==type('str') or type(x).__name__=='DataFrame':
#         if type(x)==type('str'):
#             data = {'text': [x]}
#             data = pd.DataFrame(data)
#         else:
#             data = x

#         book = openpyxl.load_workbook(file_name)
#         if dt in book.sheetnames:
#             sheet = book[dt]
#             start_row = sheet.max_row + 1  # Find the first empty row
            
#         else:
#             sheet = book.create_sheet(dt)  # Create a new sheet
#             start_row = 1

#          # Convert DataFrame to rows and append to the sheet
#         for r_idx, row in enumerate(dataframe_to_rows(data, index=False, header=False), start=start_row):
#             for c_idx, value in enumerate(row, start=1):
#                 sheet.cell(row=r_idx, column=c_idx, value=value)

#         book.save(file_name)

#     elif image==1:
#         book = openpyxl.load_workbook(file_name)
#         sheet = book[dt]
#         img = Image(add)
#         sheet.add_image(img)
#         book.save(file_name)
#     else:
#         raise ValueError('what are you trying to save?')




# def plot_categorization(df, date, pred, field='close', ):
#     """ Plot categorization for a given day with dynamic field selection """
#     df_day = df.loc[date]
#     df_day['preds'] = list(pred)

#     plt.figure(figsize=(14, 7))
#     fig, axs = plt.subplots(2, 1, figsize=(14,14))
#     axs[0].plot(df_day.index, df_day[field], label=f'{field.capitalize()} Price', color='gray', linewidth=2)
#     for cat, color in zip(['A', 'B', 'C'], ['green', 'red', 'gray']):
#         axs[0].scatter(df_day[df_day['category'] == cat].index,
#                        df_day[df_day['category'] == cat][field],
#                        color=color, label=f'Category {cat}',
#                        s=30 if cat != 'C' else 0)
#     axs[0].grid(axis='x', which='major', linestyle=':', linewidth='0.5', color='gray')
#     axs[0].grid(axis='x', which='minor', linestyle=':', linewidth='0.5', color='gray')
#     axs[0].xaxis.set_minor_locator(AutoMinorLocator(n=10))

#     axs[1].plot(df_day.index, df_day[field], label=f'{field.capitalize()} Price', color='gray', linewidth=2)
#     for cat, color in zip(['A', 'B', 'C'], ['green', 'red', 'gray']):
#         axs[1].scatter(df_day[df_day['preds'] == cat].index,
#                        df_day[df_day['preds'] == cat][field],
#                        color=color, label=f'Preds {cat}',
#                        s=20 if cat != 'C' else 0)
#     axs[1].grid(axis='x', which='major', linestyle=':', linewidth='0.5', color='gray')
#     axs[1].grid(axis='x', which='minor', linestyle=':', linewidth='0.5', color='gray')
#     axs[1].xaxis.set_minor_locator(AutoMinorLocator(n=10))

#     plt.legend()
#     plt.title(f'Price Categorization on {date}')
#     plt.xlabel('Timestamp')
#     plt.ylabel(f'{field.capitalize()} Price')
#     # plt.show()
#     plt.savefig(f'{images_directory}/plot for day - {date}.png')
#     # log(None, dt, image=1, add=f'results/images/plot for day - {date}.png')
#     log("\n\n", date)