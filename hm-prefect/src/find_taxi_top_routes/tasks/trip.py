import asyncio
import io

import pandas as pd
from prefect import task
from prefect_aws import AwsCredentials
from prefect_aws.s3 import s3_download
from utils.enum import CalcMethod


@task
def union_all(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)


async def load_trips(
    credentials: AwsCredentials, data_paths: list[str]
) -> pd.DataFrame:
    data_list = await asyncio.gather(
        *[
            s3_download(
                bucket="hongbomiao-bucket",
                key=path,
                aws_credentials=credentials,
            )
            for path in data_paths
        ]
    )
    trip_dfs = [
        pd.read_parquet(io.BytesIO(data), engine="pyarrow") for data in data_list
    ]
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
