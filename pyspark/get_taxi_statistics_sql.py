import functools

from pyspark.sql import DataFrame, SparkSession


def unionAll(*dfs):
    return functools.reduce(DataFrame.unionAll, dfs)


def get_trips(df: DataFrame) -> DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    return df.toDF(*column_names)


def get_zones(df: DataFrame) -> DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    df = df.toDF(*column_names)
    return df.drop("objectid")


def get_top_routes(
    spark: SparkSession, trips: DataFrame, zones: DataFrame
) -> DataFrame:
    trips.createOrReplaceTempView("trips")
    zones.createOrReplaceTempView("zones")

    return spark.sql(
        """
        WITH t2 AS (
            WITH t1 AS (
                SELECT pulocationid, dolocationid, count(*) AS count
                FROM trips
                GROUP BY pulocationid, dolocationid
            )
            SELECT
                t1.pulocationid,
                zones.zone AS pulocation_zone,
                zones.borough AS pulocation_borough,
                t1.dolocationid,
                t1.count
            FROM t1
            INNER JOIN zones ON t1.pulocationid = zones.locationid
        )
        SELECT
            t2.pulocationid,
            t2.pulocation_zone,
            t2.pulocation_borough,
            t2.dolocationid,
            zones.zone AS dolocation_zone,
            zones.borough AS dolocation_borough,
            t2.count
        FROM t2
        INNER JOIN zones ON t2.dolocationid = zones.locationid
        ORDER BY t2.count DESC
        """
    )


def get_taxi_statistics(trip_data_paths: list[str], zone_data_path: str) -> None:
    spark = (
        SparkSession.builder.master("local")
        .appName("get_taxi_statistics_sql")
        .getOrCreate()
    )

    trip_dfs = []
    for path in trip_data_paths:
        trip_dfs.append(spark.read.parquet(path))
    trip_df = unionAll(*trip_dfs)

    zone_df = (
        spark.read.format("csv")
        .option("inferSchema", True)
        .option("header", True)
        .load(zone_data_path)
    )

    trips = get_trips(trip_df)
    print((trips.count(), len(trips.columns)))
    trips.show()

    zones = get_zones(zone_df)
    print((zones.count(), len(zones.columns)))
    zones.show()

    top_routes = get_top_routes(spark, trips, zones)
    top_routes.show(truncate=False)


if __name__ == "__main__":
    trip_data_paths = [
        "data/yellow_tripdata_2021-07.parquet",
        "data/yellow_tripdata_2021-08.parquet",
        "data/yellow_tripdata_2021-09.parquet",
        "data/yellow_tripdata_2021-10.parquet",
        "data/yellow_tripdata_2021-11.parquet",
        "data/yellow_tripdata_2021-12.parquet",
        "data/yellow_tripdata_2022-01.parquet",
        "data/yellow_tripdata_2022-02.parquet",
        "data/yellow_tripdata_2022-03.parquet",
        "data/yellow_tripdata_2022-04.parquet",
        "data/yellow_tripdata_2022-05.parquet",
        "data/yellow_tripdata_2022-06.parquet",
    ]
    zone_data_path = "data/taxi_zones.csv"
    get_taxi_statistics(trip_data_paths, zone_data_path)
