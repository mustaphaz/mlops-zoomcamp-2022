import argparse
import os
import pickle

import pandas as pd

CATEGORICAL = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year',
                        dest='year',
                        type=int,
                        help="year of the dataset"
                        )
    parser.add_argument('-m', '--month',
                        dest='month',
                        type=int,
                        help='month of the dataset'
                        )
    args = parser.parse_args()

    data_uri = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{args.year:04d}-{args.month:02d}.parquet"
    df = read_data(data_uri)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')

    df_result = pd.concat([df[['ride_id']], (pd.DataFrame(y_pred, columns=['pred']))], axis=1)

    output_file = f"./output/{args.year:04d}_{args.month:02d}_output.parquet"

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(f"dataset {args.year}_{args.month}: ")
    file_size = os.path.getsize(output_file)
    print(f"Filesize: {str(round(file_size / (1024 * 1024), 3))} MB")
    print(f"Mean prediction: {y_pred.mean()}")
    return y_pred


main()
