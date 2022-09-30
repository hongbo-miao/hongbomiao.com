import asyncio

from prefect_aws.credentials import AwsCredentials
from prefect_aws.s3 import s3_download
from pydantic import BaseModel, validator
from tasks.route import get_top_routes, print_routes
from tasks.trip import (
    get_average_amount,
    get_average_trip_distance,
    get_is_higher_than_unit_price,
    get_price_per_mile,
    get_trip,
    get_trips,
)
from tasks.zone import get_zones
from utils.enum import CalcMethod

from prefect import flow, get_run_logger


class Model(BaseModel):
    calc_method: CalcMethod
    trip_id: int
    top_count: int

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
    credentials = await AwsCredentials.load("aws-credentials-block")
    trip_data = await s3_download(
        bucket="hongbomiao-bucket",
        key="taxi/m6nq-qud6.csv",
        aws_credentials=credentials,
    )
    trips = get_trips(trip_data)

    zone_data = await s3_download(
        bucket="hongbomiao-bucket",
        key="taxi/taxi_zones.csv",
        aws_credentials=credentials,
    )
    zones = get_zones(zone_data)

    average_trip_distance = get_average_trip_distance(trips, model.calc_method)
    logger.info(average_trip_distance)

    average_amount = get_average_amount(trips, model.calc_method)
    logger.info(average_amount)

    price_per_mile = get_price_per_mile(average_trip_distance, average_amount)
    logger.info(price_per_mile)

    trip = get_trip(model.trip_id, trips)
    is_higher_than_average_amount = get_is_higher_than_unit_price(trip, average_amount)
    logger.info(is_higher_than_average_amount)

    top_routes = get_top_routes(trips, model.top_count)
    logger.info(top_routes)

    print_routes(top_routes, zones)


if __name__ == "__main__":
    external_model = Model(calc_method=CalcMethod.MEDIAN, trip_id=42, top_count=10)
    asyncio.run(get_taxi_statistics(external_model))
