from utils.trip import load_trips, preprocess_trips
from utils.zone import load_zones, preprocess_zones

from pyspark.sql import DataFrame, SparkSession


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
        SparkSession.builder.master("local[*]")
        .appName("get_taxi_statistics_sql")
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

    top_routes = get_top_routes(spark, trips, zones)
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
