import pickle
from datetime import datetime

import mlflow
import pandas as pd
from dateutil.relativedelta import relativedelta
from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "prefect-demo"


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")

    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(date_str):
    logger = get_run_logger()

    if date_str is None:
        date = datetime.now()
    else:
        date = datetime.strptime(date_str, '%Y-%m-%d')

    training_month = (date - relativedelta(months=2)).month
    validation_month = (date - relativedelta(months=1)).month

    train_path = f'./data/fhv_tripdata_2021-{training_month:02d}.parquet'
    val_path = f'./data/fhv_tripdata_2021-{validation_month:02d}.parquet'
    logger.info(f'Train path: {train_path}')
    logger.info(f'Validation path: {val_path}')
    return train_path, val_path


@flow()
def main(date="2021-08-15"):
    train_path, val_path = get_paths(date).result()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f"output/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f"output/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)


main()

DeploymentSpec(
    flow=main,
    name='model_training',
    schedule=CronSchedule(cron='0 9 15 * *', timezone="Europe/Amsterdam"),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)

