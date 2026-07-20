import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging
import dagshub

dagshub.init(repo_owner='Kaushal-1508', repo_name='MLFLOW_foundation', mlflow=True)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    return rmse, mae, r2, accuracy

def get_or_create_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    X ,y = datasets.load_iris(return_X_y=True)
    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Use the train/test split produced above
    train_x = X_train
    test_x = X_test
    train_y = y_train
    test_y = y_test

    params = {"penalty": sys.argv[1] if len(sys.argv) > 1 else 'l2',
              "solver": sys.argv[2] if len(sys.argv) > 2 else 'lbfgs',
              "multi_class": sys.argv[3] if len(sys.argv) > 3 else 'auto',
              "random_state": 42,
              "max_iter": int(sys.argv[5]) if len(sys.argv) > 5 else 100
              }
    experiment_id = get_or_create_experiment_id("LogisticRegressionIrisModel")
    with mlflow.start_run(experiment_id=experiment_id):
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_test)

        (rmse, mae, r2, accuracy) = eval_metrics(y_test, predicted_qualities)

        print("Logistic Regression model ( penalty=%s, solver=%s, multi_class=%s, max_iter=%d):" % (params["penalty"], params["solver"], params["multi_class"], params["max_iter"]))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  Accuracy: %s" % accuracy)

        
        mlflow.log_param("penalty", params["penalty"])
        mlflow.log_param("solver", params["solver"])
        mlflow.log_param("multi_class", params["multi_class"])
        mlflow.log_param("max_iter", params["max_iter"])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.set_tag("training info", "LogisticRegressionIrisModel")
        signature = infer_signature(X_train, lr.predict(X_train))
        model_info = mlflow.sklearn.log_model(sk_model=lr, artifact_path="model", signature=signature, input_example=X_train, registered_model_name="LogisticRegressionIrisModel")

        remote_server_uri = "https://dagshub.com/Kaushal-1508/MLFLOW_foundation.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, name="model",registered_model_name="LogisticRegressionIrisModel")
        else:
            mlflow.sklearn.log_model(lr, name="model")