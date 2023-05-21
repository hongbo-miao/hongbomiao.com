import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from prefect import task
from prefect_aws import AwsCredentials


@task
async def write_to_delta_lake(df: pd.DataFrame, delta_table_path: str) -> None:
    aws_credentials = await AwsCredentials.load("ingest-data-aws-credentials-block")
    storage_options = {
        "AWS_DEFAULT_REGION": aws_credentials.region_name,
        "AWS_ACCESS_KEY_ID": aws_credentials.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_credentials.aws_secret_access_key.get_secret_value(),
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    delta_table = DeltaTable(
        table_uri=delta_table_path, storage_options=storage_options
    )
    schema = pa.schema(
        [
            ("timestamp", pa.float64()),
            ("current", pa.float64()),
            ("voltage", pa.float64()),
            ("temperature", pa.float64()),
        ]
    )
    write_deltalake(
        delta_table,
        df,
        mode="append",
        schema=schema,
        storage_options=storage_options,
    )
