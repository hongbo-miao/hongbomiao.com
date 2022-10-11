import io

import pandas as pd
from prefect_aws.s3 import s3_download
from utils.enum import CalcMethod

from prefect import task


@task
def union_all(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)


async def load_trips(credentials, data_paths):
    trip_dfs = []
    for path in data_paths:
        trip_data = await s3_download(
            bucket="hongbomiao-bucket",
            key=path,
            aws_credentials=credentials,
        )
        df = pd.read_parquet(io.BytesIO(trip_data), engine="pyarrow")
        trip_dfs.append(df)

    return union_all(*trip_dfs)


@task
def preprocess_trips(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(str.lower, axis="columns")


@task
def get_average_amount(trips: pd.DataFrame, calc_method: CalcMethod) -> float:
    match calc_method:
        case CalcMethod.MEAN:
            return trips["total_amount"].mean()
        case CalcMethod.MEDIAN:
            return trips["total_amount"].median()
        case _:
            raise ValueError(f"Unknown calc_method: {calc_method}")


@task
def get_average_trip_distance(trips: pd.DataFrame, calc_method: CalcMethod) -> float:
    match calc_method:
        case CalcMethod.MEAN:
            return trips["trip_distance"].mean()
        case CalcMethod.MEDIAN:
            return trips["trip_distance"].median()
        case _:
            raise ValueError(f"Unknown calc_method: {calc_method}")


@task
def get_price_per_mile(trip_distance: float, amount: float) -> float:
    return amount / trip_distance
