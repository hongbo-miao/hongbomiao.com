from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc
from utils.trip import load_trips, preprocess_trips
from utils.zone import load_zones, preprocess_zones


def main(data_dirname: str, trip_filenames: list[str], zone_filename: str) -> None:
    trip_data_paths = [f"{data_dirname}/{f}" for f in trip_filenames]
    zone_data_path = f"{data_dirname}/{zone_filename}"

    spark = SparkSession.builder.getOrCreate()

    trips = load_trips(spark, trip_data_paths)
    zones = load_zones(spark, zone_data_path)

    trips = preprocess_trips(trips)
    print((trips.count(), len(trips.columns)))
    trips.show()

    zones = preprocess_zones(zones)
    print((zones.count(), len(zones.columns)))
    zones.show()

    top_routes = (
        trips.select("pulocationid", "dolocationid")
        .groupBy("pulocationid", "dolocationid")
        .agg(count("*").alias("total"))
        .join(zones, on=[trips.pulocationid == zones.locationid], how="inner")
        .withColumnRenamed("zone", "pulocation_zone")
        .withColumnRenamed("borough", "pulocation_borough")
        .drop("locationid", "shape_area", "shape_leng", "the_geom")
        .join(zones, on=[trips.dolocationid == zones.locationid], how="inner")
        .withColumnRenamed("zone", "dolocation_zone")
        .withColumnRenamed("borough", "dolocation_borough")
        .drop("locationid", "shape_area", "shape_leng", "the_geom")
        .orderBy(desc("total"))
    )
    print((top_routes.count(), len(top_routes.columns)))
    top_routes.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    # https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    external_data_dirname = "data"
    external_trip_filenames = [
        "yellow_tripdata_2022-01.parquet",
        "yellow_tripdata_2022-02.parquet",
        "yellow_tripdata_2022-03.parquet",
        "yellow_tripdata_2022-04.parquet",
        "yellow_tripdata_2022-05.parquet",
        "yellow_tripdata_2022-06.parquet",
    ]
    external_zone_filename = "taxi_zones.csv"
    main(external_data_dirname, external_trip_filenames, external_zone_filename)
