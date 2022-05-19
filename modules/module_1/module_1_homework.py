import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

TRAIN_SET = 'fhv_tripdata_2021-01.parquet'
VALIDATION_SET = 'fhv_tripdata_2021-02.parquet'
DATA_DIR = './data/'


def load(filename: str) -> DataFrame:
    return pd.read_parquet(DATA_DIR + filename)


def preprocess(df: DataFrame) -> DataFrame:
    # calculate duration and add as columns
    df['duration'] = df['dropOff_datetime'] - df['pickup_datetime']
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

    df['PUlocationID'] = df['PUlocationID'].fillna(-1)
    df['DOlocationID'] = df['DOlocationID'].fillna(-1)

    rows_before_dropping = df.shape[0]
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    rows_after_dropping = df.shape[0]
    print(f'Dropped {rows_before_dropping - rows_after_dropping} rows')

    missing_values = (df['PUlocationID'] == -1).sum() / df['PUlocationID'].count()
    print('missing_values', missing_values)
    return df

def train_model(df: DataFrame) -> (DictVectorizer, LinearRegression):
    train_dicts = df_train[['PUlocationID', 'DOlocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return X_train, dv, lr

if __name__ == '__main__':
    df_train = preprocess(load(TRAIN_SET))
    X_train, dv, lr = train_model(df_train)
    y_train = df_train['duration'].values

    y_train_pred = lr.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

    df_val = preprocess(load(VALIDATION_SET))
    val_dicts = df_val[['PUlocationID', 'DOlocationID']].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_val = df_val['duration'].values

    y_val_pred = lr.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

    print('rmse_train:', rmse_train)
    print('rmse_val:', rmse_val)
