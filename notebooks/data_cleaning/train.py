import os
import pandas as pd
import json
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(args):
    # Assuming the environment variables are set by the SageMaker XGBoost estimator
    print('hereee')
    print(args)
    # test_date = os.environ['test_date']
    test_date = args.test_date
    data_path = os.environ['SM_CHANNEL_TRAIN_FEATURES']
    # train_labels_path = os.environ['SM_CHANNEL_TRAIN_LABELS']

    # Load the datasets
    data = pd.read_parquet(data_path).fillna(0)
    train_X = data.loc[:test_date, ].drop(columns='category')
    train_y = data.loc[:test_date, 'category'] # ['category']

    test_X = data.loc[test_date:, ].drop(columns='category')
    test_y = data.loc[test_date:, 'category'] # ['category']

    return train_X, train_y, test_X, test_y 

def train_model(train_X, train_y, test_X, test_y, args):
    # Splitting the dataset for training and validation
    # X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    # Creating DMatrix for XGBoost
    dtrain = xgb.DMatrix(data=train_X, label=train_y)
    # dval = xgb.DMatrix(data=X_val, label=y_val)
    dtest = xgb.DMatrix(data=test_X, label=test_y)

    # Parameters for XGBoost
    # params = os.environ['hyperparameters']
    params = {
        'objective': args.objective,
        'num_class': args.num_class,                  # Since you have three classes A, B, C
    }
    num_boost_round = args.num_round

    # Training the model
    bst = xgb.train(params, dtrain, num_boost_round, evals=[(dtest, 'test')], early_stopping_rounds=10)

    return bst

def evaluate_model(model, X_val, y_val):
    # Making predictions
    dval = xgb.DMatrix(X_val)
    predictions = model.predict(dval)

    # Calculating accuracy
    accuracy = accuracy_score(y_val, predictions)
    print(f'Validation Accuracy: {accuracy}')

def save_model(model, model_path):
    model.save_model(model_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', type=str, default='multi:softmax',
                        help='objective function')
    parser.add_argument('--num_class', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--num_round', type=int, default=100,
                        help='number of boosting rounds')
    parser.add_argument('--test_date', type=str, default='2024-10-01',
                        help='test date cut off')

    # Parse arguments
    args = parser.parse_args()
    train_X, train_y, test_X, test_y = load_data(args)
    model = train_model(train_X, train_y, test_X, test_y, args)  # Adjust based on your label column name
    evaluate_model(model, test_X, test_y)  # Adjust based on your label column name
    save_model(model, os.environ['SM_MODEL_DIR'] + '/xgboost-model')
