import functools

from pyspark.sql import DataFrame


def union_all(*dfs: DataFrame) -> DataFrame:
    return functools.reduce(DataFrame.unionAll, dfs)
