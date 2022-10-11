import asyncio
import io

import pandas as pd
from prefect_aws.credentials import AwsCredentials
from prefect_aws.s3 import s3_download
from pydantic import BaseModel, validator
from tasks.route import get_top_routes
from tasks.trip import (
    get_average_amount,
    get_average_trip_distance,
    get_price_per_mile,
    load_trips,
    preprocess_trips,
)
from tasks.zone import preprocess_zones
from utils.enum import CalcMethod

from prefect import flow, get_run_logger


class Model(BaseModel):
    calc_method: CalcMethod
    trip_id: int

    @validator("calc_method")
    def calc_method_must_be_in_enum(cls, v: CalcMethod) -> CalcMethod:
        if v not in CalcMethod:
            raise ValueError(f"calc_method must be in {CalcMethod}")
        return v

    @validator("trip_id")
    def trip_id_must_be_in_range(cls, v: int) -> int:
        if 0 <= v < 100:
            return v
        raise ValueError("trip_id must be in range [0, 100)")


@flow
async def get_taxi_statistics(model: Model) -> None:
    logger = get_run_logger()

    dirname = "taxi"
    trip_filenames = [
        "yellow_tripdata_2021-07.parquet",
        "yellow_tripdata_2021-08.parquet",
        "yellow_tripdata_2021-09.parquet",
        "yellow_tripdata_2021-10.parquet",
        "yellow_tripdata_2021-11.parquet",
        "yellow_tripdata_2021-12.parquet",
        "yellow_tripdata_2022-01.parquet",
        "yellow_tripdata_2022-02.parquet",
        "yellow_tripdata_2022-03.parquet",
        "yellow_tripdata_2022-04.parquet",
        "yellow_tripdata_2022-05.parquet",
        "yellow_tripdata_2022-06.parquet",
    ]
    zone_filename = "taxi_zones.csv"

    trip_data_paths = [f"{dirname}/{f}" for f in trip_filenames]
    zone_data_path = f"{dirname}/{zone_filename}"

    credentials = await AwsCredentials.load("aws-credentials-block")

    trips = await load_trips(credentials, trip_data_paths)
    trips = preprocess_trips(trips)

    zone_data = await s3_download(
        bucket="hongbomiao-bucket",
        key=zone_data_path,
        aws_credentials=credentials,
    )
    zones = pd.read_csv(io.BytesIO(zone_data))
    zones = preprocess_zones(zones)

    average_trip_distance = get_average_trip_distance(trips, model.calc_method)
    logger.info(average_trip_distance)

    average_amount = get_average_amount(trips, model.calc_method)
    logger.info(average_amount)

    price_per_mile = get_price_per_mile(average_trip_distance, average_amount)
    logger.info(price_per_mile)

    top_routes = get_top_routes(trips, zones)
    logger.info(top_routes)


if __name__ == "__main__":
    external_model = Model(calc_method=CalcMethod.MEDIAN, trip_id=42)
    asyncio.run(get_taxi_statistics(external_model))
