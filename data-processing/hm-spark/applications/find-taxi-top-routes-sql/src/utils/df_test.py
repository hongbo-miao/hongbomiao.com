from df import union_all
from pyspark.sql import SparkSession


class TestUnionAll:
    def test_with_two_dataframes(self) -> None:
        spark = SparkSession.builder.getOrCreate()
        df1 = spark.createDataFrame([(1, "a"), (2, "b")], ["col1", "col2"])
        df2 = spark.createDataFrame([(3, "c"), (4, "d")], ["col1", "col2"])
        df = union_all(df1, df2)
        expected_df = spark.createDataFrame(
            [(1, "a"), (2, "b"), (3, "c"), (4, "d")],
            ["col1", "col2"],
        )
        assert df.collect() == expected_df.collect()
