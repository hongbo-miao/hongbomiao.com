import asyncio
import io
import json
from pathlib import Path

import pandas as pd
from influxdb_client import Point, WritePrecision
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client.client.write_api_async import WriteApiAsync
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret
from prefect.tasks import exponential_backoff
from prefect_aws.credentials import AwsCredentials
from prefect_aws.s3 import s3_download
from tenacity import retry, stop_after_attempt, wait_random_exponential


def log_attempt_number(retry_state):
    logger = get_run_logger()
    logger.warning(f"Retrying {retry_state.attempt_number}...")


@retry(
    wait=wait_random_exponential(multiplier=40, max=120),
    stop=stop_after_attempt(60),
    after=log_attempt_number,
)
async def write_points(
    influxdb_write_api: WriteApiAsync,
    influxdb_bucket: str,
    influxdb_org: str,
    points: list[Point],
) -> None:
    logger = get_run_logger()
    try:
        await influxdb_write_api.write(influxdb_bucket, influxdb_org, points)
    except Exception as e:
        error_message = str(e)
        if "timeout" in error_message:
            logger.warning(f"WARN-001 | {error_message}")
            raise e
        else:
            logger.error(f"ERR-001 | {error_message}")
            raise e


@task(
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    retry_jitter_factor=1,
)
async def write_to_influxdb(
    data: bytes,
    influxdb_url: str,
    influxdb_org: str,
    influxdb_bucket: str,
) -> None:
    logger = get_run_logger()

    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    df = df.rename(str.lower, axis="columns")

    influxdb_token_block = await Secret.load("influxdb-token-block")
    async with InfluxDBClientAsync(
        url=influxdb_url,
        org=influxdb_org,
        token=influxdb_token_block.get(),
        enable_gzip=True,
        timeout=60000,  # ms
    ) as influxdb_client:
        influxdb_write_api = influxdb_client.write_api()
        count = 0
        points = []
        for row in df.itertuples():
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
            points.append(point)
            count += 1
            if len(points) == 2000:
                await write_points(
                    influxdb_write_api, influxdb_bucket, influxdb_org, points
                )
                points = []
                logger.info(f"Wrote {count} points.")

        if len(points) > 0:
            await write_points(
                influxdb_write_api, influxdb_bucket, influxdb_org, points
            )
            logger.info(f"Wrote {count} points.")

        logger.info(f"Finished writing {count} points.")


@flow
async def upload_to_influxdb(
    influxdb_url: str, prefect_tags: list[str], trip_data_paths: list[str]
) -> None:
    influxdb_org = "hongbomiao"
    influxdb_bucket = "hm-taxi-bucket"
    credentials = await AwsCredentials.load("upload-to-influxdb-aws-credentials-block")

    for trip_data_path in trip_data_paths:
        filename = Path(trip_data_path).name
        s3_download_with_options = s3_download.with_options(name=f"download-{filename}")
        data = await s3_download_with_options(
            bucket="hongbomiao-bucket",
            key=trip_data_path,
            aws_credentials=credentials,
        )

        write_to_influxdb_with_options = write_to_influxdb.with_options(
            name=f"write-{filename}-to-influxdb", tags=prefect_tags
        )
        await write_to_influxdb_with_options(
            data, influxdb_url, influxdb_org, influxdb_bucket
        )


if __name__ == "__main__":
    params = json.loads(Path("params.json").read_text())
    external_influxdb_url = "http://localhost:20622"
    asyncio.run(
        upload_to_influxdb(
            influxdb_url=external_influxdb_url,
            prefect_tags=params["prefect_tags"],
            trip_data_paths=params["trip_data_paths"],
        )
    )
