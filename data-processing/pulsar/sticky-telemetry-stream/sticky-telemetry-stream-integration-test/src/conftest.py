import os

import boto3
import pytest
from botocore.client import BaseClient


@pytest.fixture
def postgres_url() -> str:
    return os.environ["TEST_POSTGRES_URL"]


@pytest.fixture
def s3_client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=os.environ["TEST_RUSTFS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["TEST_RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TEST_RUSTFS_SECRET_KEY"],
        region_name="us-west-2",
    )
