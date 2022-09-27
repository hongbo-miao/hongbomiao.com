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
    postive_df = df[mask]
    return postive_df


@flow
async def fetch_taxi_data():
    logger = get_run_logger()
    aws_credentials_block = await AwsCredentials.load("aws-credentials-block")
    credentials = AwsCredentials(
        aws_access_key_id=aws_credentials_block.aws_access_key_id,
        aws_secret_access_key=aws_credentials_block.aws_secret_access_key,
    )
    data = await s3_download(
        bucket="hongbomiao-bucket",
        key="hm-airflow/taxi.csv",
        aws_credentials=credentials,
    )
    df = get_dataframe(data)
    logger.info(df)


if __name__ == "__main__":
    asyncio.run(fetch_taxi_data())
