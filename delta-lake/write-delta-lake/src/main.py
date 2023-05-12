import pandas as pd
import pyarrow as pa
from deltalake.writer import write_deltalake

df = pd.read_parquet("data/motor.parquet", engine="pyarrow")
s3_path = "s3a://hongbomiao-bucket/delta-tables/motor"
storage_options = {
    "AWS_DEFAULT_REGION": "us-west-2",
    "AWS_ACCESS_KEY_ID": "xxx",
    "AWS_SECRET_ACCESS_KEY": "xxx",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}
schema = pa.schema(
    [
        ("timestamp", pa.float64()),
        ("current", pa.float64()),
        ("voltage", pa.float64()),
        ("temperature", pa.float64()),
    ]
)
write_deltalake(
    s3_path,
    df,
    mode="append",
    schema=schema,
    storage_options=storage_options,
)
