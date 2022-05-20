import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from modules.module_1.model_trainer import ModelTrainer

TRAIN_SET = 'fhv_tripdata_2021-01.parquet'
VALIDATION_SET = 'fhv_tripdata_2021-02.parquet'
DATA_DIR = './data/'
FEATURES = ['PUlocationID', 'DOlocationID']
TARGET = 'duration'


def _load(filename: str) -> DataFrame:
    return pd.read_parquet(DATA_DIR + filename)


def _preprocess(df: DataFrame) -> (dict, pd.Series):
    _df = df.copy()
    _df = _add_duration_column(_df)
    _df = _impute_missing_values(_df)
    _df_filtered = _filter_duration_between_1_and_60_min(_df)
    transformed_features = _df_filtered[FEATURES].to_dict(orient='records')
    target = _df_filtered[TARGET].values
    return transformed_features, target


def _filter_duration_between_1_and_60_min(_df):
    return _df[(_df['duration'] >= 1) & (_df['duration'] <= 60)]


def _impute_missing_values(_df: DataFrame) -> DataFrame:
    _df['PUlocationID'] = _df['PUlocationID'].fillna(-1)
    _df['DOlocationID'] = _df['DOlocationID'].fillna(-1)
    return _df


def _add_duration_column(_df: DataFrame) -> DataFrame:
    _df['duration'] = _df['dropOff_datetime'] - _df['pickup_datetime']
    _df['duration'] = _df['duration'].apply(lambda td: td.total_seconds() / 60)
    return _df


def train() -> Pipeline:
    training_data = _load(TRAIN_SET)
    transformed_features, target = _preprocess(training_data)
    trainer = ModelTrainer(features=transformed_features, target=target)
    pipeline = trainer.train_model()
    return pipeline


def validate(model: Pipeline, X: dict, y: pd.Series):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred, squared=False)


if __name__ == '__main__':
    model = train()
    train_data = _load(TRAIN_SET)
    validation_data = _load(VALIDATION_SET)

    X_train, y_train = _preprocess(train_data)
    X_validate, y_validate = _preprocess(validation_data)

    rmse_train = validate(model, X_train, y_train)
    rmse_validate = validate(model, X_validate, y_validate)

    print(f"RMSE on train: {rmse_train}")
    print(f"RMSE on validate: {rmse_validate}")
