import asyncio
import io

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, WriteApi
from prefect import flow, task
from prefect_aws.credentials import AwsCredentials
from prefect_aws.s3 import s3_download


@task
def write_to_influxdb(
    row, influxdb_org: str, influxdb_bucket: str, influxdb_write_api: WriteApi
):
    point = (
        Point("trip")
        .tag("vendorid", row.vendorid)
        .field("tpep_pickup_datetime", row.tpep_pickup_datetime.isoformat())
        .field("tpep_dropoff_datetime", row.tpep_dropoff_datetime.isoformat())
        .field("passenger_count", row.passenger_count)
        .field("trip_distance", row.trip_distance)
        .field("ratecodeid", row.ratecodeid)
        .field("store_and_fwd_flag", row.store_and_fwd_flag)
        .field("store_and_fwd_flag", row.store_and_fwd_flag)
        .field("pulocationid", row.pulocationid)
        .field("dolocationid", row.dolocationid)
        .field("payment_type", row.payment_type)
        .field("fare_amount", row.fare_amount)
        .field("extra", row.extra)
        .field("mta_tax", row.mta_tax)
        .field("tip_amount", row.tip_amount)
        .field("tolls_amount", row.tolls_amount)
        .field("improvement_surcharge", row.improvement_surcharge)
        .field("total_amount", row.total_amount)
        .field("congestion_surcharge", row.congestion_surcharge)
        .field("airport_fee", row.airport_fee)
        .time(row.tpep_pickup_datetime, WritePrecision.NS)
    )

    influxdb_write_api.write(influxdb_bucket, influxdb_org, point)


async def process(
    path: str, influxdb_org: str, influxdb_bucket: str, influxdb_write_api: WriteApi
) -> None:
    credentials = await AwsCredentials.load("aws-credentials-block")
    data = await s3_download(
        bucket="hongbomiao-bucket",
        key=path,
        aws_credentials=credentials,
    )
    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    df = df.rename(str.lower, axis="columns")
    for row in df.itertuples():
        write_to_influxdb(row, influxdb_org, influxdb_bucket, influxdb_write_api)


@flow
async def find_taxi_top_routes() -> None:
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

    trip_data_paths = [f"{dirname}/{f}" for f in trip_filenames]

    influxdb_token = ""
    influxdb_org = "hongbomiao"
    influxdb_bucket = "hm-taxi-bucket"

    with InfluxDBClient(
        url="http://localhost:20622", token=influxdb_token, org=influxdb_org
    ) as client:
        influxdb_write_api = client.write_api(write_options=SYNCHRONOUS)
        for path in trip_data_paths:
            await process(path, influxdb_org, influxdb_bucket, influxdb_write_api)


if __name__ == "__main__":
    asyncio.run(find_taxi_top_routes())
