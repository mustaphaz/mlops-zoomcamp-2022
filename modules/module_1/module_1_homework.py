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
ANSWERS = {}


def _load_data(filename: str) -> DataFrame:
    _df = pd.read_parquet(DATA_DIR + filename)
    ANSWERS["Q1: Read the data for January. How many records are there?"] = len(_df)
    _df['duration'] = _df['dropOff_datetime'] - _df['pickup_datetime']
    _df['duration'] = _df['duration'].dt.total_seconds() / 60
    ANSWERS["Q2: What's the average trip duration in January?"] = _df['duration'].mean()
    _df = _df[(_df['duration'] >= 1) & (_df['duration'] <= 60)].copy()
    ANSWERS["Q3: What's the fractions of missing values for the pickup location ID? "] = \
        _df['PUlocationID'].isnull().mean()
    _df[FEATURES] = _df[FEATURES].fillna(-1).astype('int').astype('str')
    return _df


def train(_df: DataFrame) -> Pipeline:
    feature_dicts = _df[FEATURES].to_dict(orient='records')
    target = _df[TARGET].values
    trainer = ModelTrainer(features=feature_dicts, target=target)
    return trainer.train_model()


def validate(model: Pipeline, _df: DataFrame) -> float:
    X = _df[FEATURES].to_dict(orient='records')
    y = _df[TARGET].values
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred, squared=False)


if __name__ == '__main__':
    train_data = _load_data(TRAIN_SET)
    validation_data = _load_data(VALIDATION_SET)

    model = train(train_data)

    ANSWERS["Q4: What's the dimensionality of this matrix?"] = len(model['vect'].feature_names_)
    ANSWERS["Q5: What's the RMSE on train?"] = validate(model, train_data)
    ANSWERS["Q5: What's the RMSE on validation?"] = validate(model, validation_data)

    for k, v in ANSWERS.items():
        print(f'{k} \n\t{v}')
