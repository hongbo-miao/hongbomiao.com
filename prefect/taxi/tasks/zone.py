import io

import pandas as pd

from prefect import task


@task
def get_zones(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    df = df.rename(str.lower, axis="columns")
    df = df.set_index("locationid")
    df["borough"] = df["borough"].astype("category")
    return df.drop(columns=["objectid"])
