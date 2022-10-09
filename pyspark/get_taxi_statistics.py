from pyspark.sql import SparkSession, dataframe
from pyspark.sql.functions import count, desc


def get_trips(df: dataframe.DataFrame) -> dataframe.DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    return df.toDF(*column_names)


def get_zones(df: dataframe.DataFrame) -> dataframe.DataFrame:
    column_names = list(map(lambda x: x.lower(), df.columns))
    df = df.toDF(*column_names)
    return df.drop("objectid")


def get_top_routes(
    trips: dataframe.DataFrame, zones: dataframe.DataFrame
) -> dataframe.DataFrame:
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


def get_taxi_statistics(taxi_data_path: str, zone_data_path: str) -> None:
    spark = SparkSession.builder.appName("hm_spark_app").getOrCreate()
    taxi_df = (
        spark.read.format("csv")
        .option("inferSchema", True)
        .option("header", True)
        .load(taxi_data_path)
    )
    zone_df = (
        spark.read.format("csv")
        .option("inferSchema", True)
        .option("header", True)
        .load(zone_data_path)
    )
    trips = get_trips(taxi_df)
    zones = get_zones(zone_df)

    print((trips.count(), len(trips.columns)))
    trips.show()

    print((zones.count(), len(zones.columns)))
    zones.show()

    top_routes = get_top_routes(trips, zones)
    top_routes.show(truncate=False)


if __name__ == "__main__":
    taxi_data_path = "data/2021_Yellow_Taxi_Trip_Data.csv"
    zone_data_path = "data/taxi_zones.csv"
    get_taxi_statistics(taxi_data_path, zone_data_path)
