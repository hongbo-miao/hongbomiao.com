import logging

from pyspark.sql import SparkSession
from utils.trip import load_trips, preprocess_trips
from utils.zone import load_zones, preprocess_zones

logger = logging.getLogger(__name__)


def main(
    data_dirname: str,
    trip_filenames: list[str],
    zone_filename: str,
) -> None:
    trip_data_paths = [f"{data_dirname}/{f}" for f in trip_filenames]
    zone_data_path = f"{data_dirname}/{zone_filename}"

    spark = SparkSession.builder.getOrCreate()

    trips = load_trips(spark, trip_data_paths)
    zones = load_zones(spark, zone_data_path)

    trips = preprocess_trips(trips)
    logger.info((trips.count(), len(trips.columns)))
    trips.show()

    zones = preprocess_zones(zones)
    logger.info((zones.count(), len(zones.columns)))
    zones.show()

    # Get top routes
    trips.createOrReplaceTempView("trips")
    zones.createOrReplaceTempView("zones")
    top_routes = spark.sql(
        """
        with t2 as (
            with t1 as (
                select
                    pulocationid,
                    dolocationid,
                    count(*) as total
                from trips
                group by pulocationid, dolocationid
            )

            select
                t1.pulocationid,
                zones.zone as pulocation_zone,
                zones.borough as pulocation_borough,
                t1.dolocationid,
                t1.total
            from t1
            inner join zones on t1.pulocationid = zones.locationid
        )

        select
            t2.pulocationid,
            t2.pulocation_zone,
            t2.pulocation_borough,
            t2.dolocationid,
            zones.zone as dolocation_zone,
            zones.borough as dolocation_borough,
            t2.total
        from t2
        inner join zones on t2.dolocationid = zones.locationid
        order by t2.total desc
        """,
    )
    logger.info((top_routes.count(), len(top_routes.columns)))
    top_routes.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

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
    main(
        external_data_dirname,
        external_trip_filenames,
        external_zone_filename,
    )
