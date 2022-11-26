import pandas as pd
from prefect import task


@task
def preprocess_zones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(str.lower, axis="columns")
    df["borough"] = df["borough"].astype("category")
    return df.drop(columns=["objectid"])
