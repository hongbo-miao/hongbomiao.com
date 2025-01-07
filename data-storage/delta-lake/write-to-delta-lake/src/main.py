import time

import config
import numpy as np
import pandas as pd
import pyarrow as pa
from deltalake.writer import write_deltalake


def main(row_count: int):
    generator = np.random.default_rng(42)
    timestamp = np.array([time.time() + i * 0.01 for i in range(row_count)])
    current = generator.standard_normal(row_count) * 10.0
    voltage = generator.standard_normal(row_count) * 20.0
    temperature = generator.standard_normal(row_count) * 50.0 + 25.0
    data = {
        "timestamp": timestamp,
        "current": current,
        "voltage": voltage,
        "temperature": temperature,
    }
    df = pd.DataFrame(data)
    storage_options = {
        "AWS_DEFAULT_REGION": config.aws_default_region,
        "AWS_ACCESS_KEY_ID": config.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": config.aws_secret_access_key,
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    schema = pa.schema(
        [
            ("timestamp", pa.float64()),
            ("current", pa.float64()),
            ("voltage", pa.float64()),
            ("temperature", pa.float64()),
        ],
    )
    write_deltalake(
        config.s3_path,
        df,
        mode="append",
        schema=schema,
        storage_options=storage_options,
    )


if __name__ == "__main__":
    external_row_count = 10
    main(external_row_count)
