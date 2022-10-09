import functools

from pyspark.sql import DataFrame


def unionAll(*dfs):
    return functools.reduce(DataFrame.unionAll, dfs)
