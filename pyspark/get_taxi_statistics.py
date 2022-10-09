from utils.trip import load_trips, preprocess_trips
from utils.zone import load_zones, preprocess_zones

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import count, desc


def get_top_routes(trips: DataFrame, zones: DataFrame) -> DataFrame:
    return (
        trips.select("pulocationid", "dolocationid")
        .groupBy("pulocationid", "dolocationid")
        .agg(count("*").alias("count"))
        .join(zones, on=[trips.pulocationid == zones.locationid], how="inner")
        .withColumnRenamed("zone", "pulocation_zone")
        .withColumnRenamed("borough", "pulocation_borough")
        .select(
            "pulocationid",
            "pulocation_zone",
            "pulocation_borough",
            "dolocationid",
            "count",
        )
        .join(zones, on=[trips.dolocationid == zones.locationid], how="inner")
        .withColumnRenamed("zone", "dolocation_zone")
        .withColumnRenamed("borough", "dolocation_borough")
        .select(
            "pulocationid",
            "pulocation_zone",
            "pulocation_borough",
            "dolocationid",
            "dolocation_zone",
            "dolocation_borough",
            "count",
        )
        .orderBy(desc("count"))
    )


def get_taxi_statistics(trip_data_paths: list[str], zone_data_path: str) -> None:
    spark = (
        SparkSession.builder.master("local")
        .appName("get_taxi_statistics")
        .getOrCreate()
    )

    trips = load_trips(spark, trip_data_paths)
    zones = load_zones(spark, zone_data_path)

    trips = preprocess_trips(trips)
    print((trips.count(), len(trips.columns)))
    trips.show()

    zones = preprocess_zones(zones)
    print((zones.count(), len(zones.columns)))
    zones.show()

    top_routes = get_top_routes(trips, zones)
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
