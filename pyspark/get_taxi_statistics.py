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
        SparkSession.builder.master("local[*]")
        .appName("get_taxi_statistics")
        .config("spark.ui.port", "4040")
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
    dirname = "data"
    trip_filenames = [
        "yellow_tripdata_2021-07.parquet",
        "yellow_tripdata_2021-08.parquet",
        "yellow_tripdata_2021-09.parquet",
        "yellow_tripdata_2021-10.parquet",
        "yellow_tripdata_2021-11.parquet",
        "yellow_tripdata_2021-12.parquet",
        "yellow_tripdata_2022-01.parquet",
        "yellow_tripdata_2022-02.parquet",
        "yellow_tripdata_2022-03.parquet",
        "yellow_tripdata_2022-04.parquet",
        "yellow_tripdata_2022-05.parquet",
        "yellow_tripdata_2022-06.parquet",
    ]
    zone_filename = "taxi_zones.csv"

    trip_data_paths = [f"{dirname}/{f}" for f in trip_filenames]
    zone_data_path = f"{dirname}/{zone_filename}"

    get_taxi_statistics(trip_data_paths, zone_data_path)
