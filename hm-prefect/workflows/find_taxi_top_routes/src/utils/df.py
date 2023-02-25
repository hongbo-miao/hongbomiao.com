import pandas as pd


def union_all(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)
