import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score , accuracy_score , roc_auc_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    URL  = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

    # read the data as df
    try:
        df=pd.read_csv(URL , sep=';')
        return df
    except Exception as e:
        raise e

def evaluate(y_true , y_pred):
    mae = mean_absolute_error(y_true , y_pred)
    mse = mean_squared_error(y_true , y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true , y_pred)
    return mae , mse , rmse , r2

def acc(y_true , y_pred , pred_prob):
    return accuracy_score(y_true , y_pred) , roc_auc_score(y_true , pred_prob , multi_class='ovr')


def main(n_estimators , max_depth):
    df = get_data()

    # train test split with raw data
    train , test = train_test_split(df)
    X_train = train.drop(['quality'] , axis=1)
    X_test = test.drop(['quality'] , axis=1)
    y_train = train[['quality']]
    y_test = test[['quality']]

    '''# Training the model
    lr = ElasticNet()
    lr.fit(X_train , y_train)
    y_pred = lr.predict(X_test)

    # Evaluate model
    mae , mse , rmse , r2 = evaluate(y_test , y_pred)
    print(f'Mean Absolute Error : {mae}')
    print(f'Mean Squared Error : {mse}')
    print(f'Root Mean Squared Error : {rmse}')
    print(f'r2 score : {r2}')'''


    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators , max_depth=max_depth)
        model.fit(X_train , y_train)
        y_pred = model.predict(X_test)
        pred_prob = model.predict_proba(X_test)

        accuracy , roc_auc= acc(y_test , y_pred , pred_prob)
        print(f"Accuracy Score  : {accuracy}")
        print(f"RocAucScore : {roc_auc}")

        # Logging param and metric with mlflow
        mlflow.log_param('n_estimators' , n_estimators)
        mlflow.log_param('max_depth' , max_depth)
        mlflow.log_metric('Accuracy' , accuracy)
        mlflow.log_metric('roc_auc_score' , roc_auc)

        # logging model with mlflow
        mlflow.sklearn.log_model(model , 'RandomForestModel')



if __name__ =='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--n_estimators' , '-n' , default=50 , type=int)
    args.add_argument('--max_depth' , '-m' , default=5 , type=int)
    parse_args = args.parse_args()

    try:
        main(n_estimators = parse_args.n_estimators , max_depth=parse_args.max_depth)
    except Exception as e:
        raise e