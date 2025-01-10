import os
from datetime import UTC, datetime
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

with DAG(
    "s3_download",
    start_date=datetime(2022, 1, 1, tzinfo=UTC),
    schedule_interval="@once",
    catchup=False,
) as dag:

    @task
    def download_from_s3(key: str, bucket: str, local_path: str) -> str:
        hook = S3Hook("s3_connection")
        file_path = hook.download_file(
            key=key,
            bucket_name=bucket,
            local_path=local_path,
        )
        return file_path

    @task
    def rename_file(file_path: str, new_file_name: str) -> None:
        path = Path(file_path)
        os.rename(src=file_path, dst=path.parent.joinpath(new_file_name))

    file_name = download_from_s3(
        bucket="hm-production-bucket",
        key="hm-airflow/taxi.csv",
        local_path="/tmp/",  # noqa: S108
    )
    rename_file(file_name, "taxi.csv")
