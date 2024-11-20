import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    # Assuming the environment variables are set by the SageMaker XGBoost estimator
    train_features_path = os.environ['SM_CHANNEL_TRAIN_FEATURES']
    train_labels_path = os.environ['SM_CHANNEL_TRAIN_LABELS']

    # Load the datasets
    train_features = pd.read_parquet(train_features_path)
    train_labels = pd.read_parquet(train_labels_path)

    return train_features, train_labels

def train_model(train_features, train_labels):
    # Splitting the dataset for training and validation
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    # Creating DMatrix for XGBoost
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)

    # Parameters for XGBoost
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.3,
        'eval_metric': 'mlogloss',
    }
    num_boost_round = 100

    # Training the model
    bst = xgb.train(params, dtrain, num_boost_round, evals=[(dval, 'validation')], early_stopping_rounds=10)

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
    train_features, train_labels = load_data()
    model = train_model(train_features, train_labels['label'])  # Adjust based on your label column name
    evaluate_model(model, train_features, train_labels['label'])  # Adjust based on your label column name
    save_model(model, os.environ['SM_MODEL_DIR'] + '/xgboost-model')
