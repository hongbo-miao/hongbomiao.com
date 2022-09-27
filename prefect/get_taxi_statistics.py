import asyncio
import io

import pandas as pd
from prefect_aws.credentials import AwsCredentials
from prefect_aws.s3 import s3_download

from prefect import flow, get_run_logger, task


@task
def get_dataframe(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(
        io.BytesIO(data), parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    )
    df = df.rename(str.lower, axis="columns")
    df["vendorid"] = df["vendorid"].astype("category")
    df["ratecodeid"] = df["ratecodeid"].astype("category")
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("category")
    df["pulocationid"] = df["pulocationid"].astype("category")
    df["dolocationid"] = df["dolocationid"].astype("category")
    df["payment_type"] = df["payment_type"].astype("category")
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("bool")
    mask = df["total_amount"] > 0
    return df[mask]


@task
def get_average_amount(df: pd.DataFrame) -> float:
    return df["total_amount"].mean()


@task
def get_average_trip_distance(df: pd.DataFrame) -> float:
    return df["trip_distance"].mean()


@task
def get_price_per_mile(trip_distance: float, amount: float) -> float:
    return amount / trip_distance


@flow
async def get_taxi_statistics() -> None:
    logger = get_run_logger()
    credentials = await AwsCredentials.load("aws-credentials-block")
    data = await s3_download(
        bucket="hongbomiao-bucket",
        key="hm-airflow/taxi.csv",
        aws_credentials=credentials,
    )
    df = get_dataframe(data)

    average_trip_distance = get_average_trip_distance(df)
    logger.info(average_trip_distance)

    average_amount = get_average_amount(df)
    logger.info(average_amount)

    price_per_mile = get_price_per_mile(average_trip_distance, average_amount)
    logger.info(price_per_mile)


if __name__ == "__main__":
    asyncio.run(get_taxi_statistics())
