import asyncio
import io
from enum import Enum

import pandas as pd
from prefect_aws.credentials import AwsCredentials
from prefect_aws.s3 import s3_download
from pydantic import BaseModel, validator

from prefect import flow, get_run_logger, task


class CalcMethod(Enum):
    MEDIAN = "median"
    MEAN = "mean"


class Model(BaseModel):
    calc_method: CalcMethod
    trip_id: int

    @validator("calc_method")
    def calc_method_must_be_in_enum(cls, v):
        if v not in CalcMethod:
            raise ValueError(f"calc_method must be in {CalcMethod}")
        return v

    @validator("trip_id")
    def trip_id_must_be_in_range(cls, v):
        if 0 <= v < 100:
            return v
        raise ValueError("trip_id must be in range [0, 100)")


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
def get_average_amount(df: pd.DataFrame, calc_method: CalcMethod) -> float:
    match calc_method:
        case CalcMethod.MEAN:
            return df["total_amount"].mean()
        case CalcMethod.MEDIAN:
            return df["total_amount"].median()
        case _:
            raise ValueError(f"Unknown calc_method: {calc_method}")


@task
def get_average_trip_distance(df: pd.DataFrame, calc_method: CalcMethod) -> float:
    match calc_method:
        case CalcMethod.MEAN:
            return df["trip_distance"].mean()
        case CalcMethod.MEDIAN:
            return df["trip_distance"].median()
        case _:
            raise ValueError(f"Unknown calc_method: {calc_method}")


@task
def get_price_per_mile(trip_distance: float, amount: float) -> float:
    return amount / trip_distance


@task
def get_trip(trip_id: int, df: pd.DataFrame) -> pd.Series:
    return df.loc[trip_id, :]


@task
def get_is_higher_than_unit_price(trip: pd.Series, average_amount: float) -> bool:
    return trip["total_amount"] / trip["trip_distance"] > average_amount


@flow
async def get_taxi_statistics(model: Model) -> None:
    logger = get_run_logger()
    credentials = await AwsCredentials.load("aws-credentials-block")
    data = await s3_download(
        bucket="hongbomiao-bucket",
        key="hm-airflow/taxi.csv",
        aws_credentials=credentials,
    )
    df = get_dataframe(data)

    average_trip_distance = get_average_trip_distance(df, model.calc_method)
    logger.info(average_trip_distance)

    average_amount = get_average_amount(df, model.calc_method)
    logger.info(average_amount)

    price_per_mile = get_price_per_mile(average_trip_distance, average_amount)
    logger.info(price_per_mile)

    trip = get_trip(model.trip_id, df)
    is_higher_than_average_amount = get_is_higher_than_unit_price(trip, average_amount)
    logger.info(is_higher_than_average_amount)


if __name__ == "__main__":
    external_model = Model(calc_method=CalcMethod.MEDIAN, trip_id=10)
    asyncio.run(get_taxi_statistics(external_model))
