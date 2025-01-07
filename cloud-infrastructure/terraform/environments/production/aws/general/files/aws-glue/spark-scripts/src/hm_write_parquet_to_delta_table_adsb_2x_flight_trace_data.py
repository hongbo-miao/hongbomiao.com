import logging
import os
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat, date_format, from_unixtime, lit, when

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

raw_parquet_paths = [
    "s3://hm-production-bucket/data/raw-parquet/adsb_2x_flight_trace_data/",
]
delta_table_path = (
    "s3://hm-production-bucket/data/delta-tables/adsb_2x_flight_trace_data/"
)
partitions = ["_date"]


def cast_column_type(df: DataFrame, column_name: str, new_data_type: str) -> DataFrame:
    if column_name in df.columns:
        df = df.withColumn(
            column_name,
            when(col(column_name).isNotNull(), col(column_name).cast(new_data_type)),
        )
    return df


def add_time_column(
    df: DataFrame,
    start_time_column_name: str,
    relative_time_column_name: str,
    time_column_name: str,
) -> DataFrame:
    return df.withColumn(
        time_column_name,
        (col(start_time_column_name) + col(relative_time_column_name))
        * lit(1000000000.0),
    ).withColumn(time_column_name, col(time_column_name).cast("bigint"))


def add_date_column(
    df: DataFrame,
    time_column_name: str,
    date_column_name: str,
) -> DataFrame:
    return df.withColumn(
        date_column_name,
        date_format(
            from_unixtime(col(time_column_name) / lit(1000000000.0)),
            "yyyy-MM-dd",
        ),
    )


def add_dbflags_columns(
    df: DataFrame,
    flag_column_name: str,
    columns_and_masks: list[tuple[str, int]],
) -> DataFrame:
    for column_name, mask in columns_and_masks:
        df = df.withColumn(
            column_name,
            when((col(flag_column_name).bitwiseAND(mask)) > 0, True).otherwise(False),
        )
    return df


def add_trace_flags_columns(
    df: DataFrame,
    flag_column_name: str,
    columns_and_masks: list[tuple[str, int]],
) -> DataFrame:
    for column_name, mask in columns_and_masks:
        if column_name in [
            "trace_flags_vertical_rate_type",
            "trace_flags_altitude_type",
        ]:
            df = df.withColumn(
                column_name,
                when(
                    (col(flag_column_name).bitwiseAND(mask)) > 0,
                    "geometric",
                ).otherwise("barometric"),
            )
        else:
            df = df.withColumn(
                column_name,
                when((col(flag_column_name).bitwiseAND(mask)) > 0, True).otherwise(
                    False,
                ),
            )
    return df


def add_trace_on_ground_column(
    df: DataFrame,
    trace_altitude_ft_column_name: str,
    trace_on_ground_column_name: str,
    ground_value: str,
) -> DataFrame:
    if trace_altitude_ft_column_name in df.columns:
        df = df.withColumn(
            trace_on_ground_column_name,
            when(
                col(trace_altitude_ft_column_name) == lit(ground_value),
                True,
            ).otherwise(False),
        )
    return df


def add_coordinate_column(
    df: DataFrame,
    longitude_column_name: str,
    latitude_column_name: str,
    coordinate_column_name: str,
) -> DataFrame:
    return df.withColumn(
        coordinate_column_name,
        concat(
            lit("POINT ("),
            col(longitude_column_name),
            lit(" "),
            col(latitude_column_name),
            lit(")"),
        ),
    )


args = getResolvedOptions(sys.argv, ["JOB_NAME"])
spark_context = SparkContext()
glue_context = GlueContext(spark_context)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

s3_node = glue_context.create_dynamic_frame.from_options(
    format_options={},
    connection_type="s3",
    format="parquet",
    connection_options={
        "paths": raw_parquet_paths,
        "recurse": True,
    },
    transformation_ctx="s3_node",
)
df = s3_node.toDF()

if df.isEmpty():
    logger.info("DataFrame is empty.")
    job.commit()
    os._exit(os.EX_OK)

# Add "trace_on_ground"
df = add_trace_on_ground_column(df, "trace_altitude_ft", "trace_on_ground", "ground")

# Convert types
df = cast_column_type(df, "dbFlags", "bigint")
df = cast_column_type(df, "desc", "string")
df = cast_column_type(df, "icao", "string")
df = cast_column_type(df, "noRegData", "boolean")
df = cast_column_type(df, "ownOp", "string")
df = cast_column_type(df, "r", "string")
df = cast_column_type(df, "t", "string")
df = cast_column_type(df, "timestamp", "double")
df = cast_column_type(df, "year", "bigint")
df = cast_column_type(df, "trace_relative_timestamp", "double")
df = cast_column_type(df, "trace_latitude_deg", "double")
df = cast_column_type(df, "trace_longitude_deg", "double")
df = cast_column_type(df, "trace_altitude_ft", "double")
df = cast_column_type(df, "trace_ground_speed_kt", "double")
df = cast_column_type(df, "trace_track_deg", "double")
df = cast_column_type(df, "trace_flags", "bigint")
df = cast_column_type(df, "trace_vertical_rate_fpm", "double")
df = cast_column_type(df, "trace_position_type", "string")
df = cast_column_type(df, "trace_geometric_altitude_ft", "double")
df = cast_column_type(df, "trace_geometric_vertical_rate_fpm", "double")
df = cast_column_type(df, "trace_indicated_airspeed_kt", "double")
df = cast_column_type(df, "trace_roll_angle_deg", "double")
df = cast_column_type(df, "trace_feeder_id", "string")
df = cast_column_type(df, "trace_aircraft_alert", "double")
df = cast_column_type(df, "trace_aircraft_alt_geom", "double")
df = cast_column_type(df, "trace_aircraft_baro_rate", "double")
df = cast_column_type(df, "trace_aircraft_category", "string")
df = cast_column_type(df, "trace_aircraft_emergency", "string")
df = cast_column_type(df, "trace_aircraft_flight", "string")
df = cast_column_type(df, "trace_aircraft_geom_rate", "double")
df = cast_column_type(df, "trace_aircraft_gva", "double")
df = cast_column_type(df, "trace_aircraft_ias", "double")
df = cast_column_type(df, "trace_aircraft_mach", "double")
df = cast_column_type(df, "trace_aircraft_mag_heading", "double")
df = cast_column_type(df, "trace_aircraft_nac_p", "double")
df = cast_column_type(df, "trace_aircraft_nac_v", "double")
df = cast_column_type(df, "trace_aircraft_nav_altitude_fms", "double")
df = cast_column_type(df, "trace_aircraft_nav_altitude_mcp", "double")
df = cast_column_type(df, "trace_aircraft_nav_heading", "double")
df = cast_column_type(df, "trace_aircraft_nav_modes", "array<string>")
df = cast_column_type(df, "trace_aircraft_nav_qnh", "double")
df = cast_column_type(df, "trace_aircraft_nic", "double")
df = cast_column_type(df, "trace_aircraft_oat", "double")
df = cast_column_type(df, "trace_aircraft_rc", "double")
df = cast_column_type(df, "trace_aircraft_roll", "double")
df = cast_column_type(df, "trace_aircraft_sda", "double")
df = cast_column_type(df, "trace_aircraft_sil", "double")
df = cast_column_type(df, "trace_aircraft_sil_type", "string")
df = cast_column_type(df, "trace_aircraft_spi", "double")
df = cast_column_type(df, "trace_aircraft_squawk", "string")
df = cast_column_type(df, "trace_aircraft_tas", "double")
df = cast_column_type(df, "trace_aircraft_tat", "double")
df = cast_column_type(df, "trace_aircraft_track", "double")
df = cast_column_type(df, "trace_aircraft_track_rate", "double")
df = cast_column_type(df, "trace_aircraft_true_heading", "double")
df = cast_column_type(df, "trace_aircraft_type", "string")
df = cast_column_type(df, "trace_aircraft_version", "double")
df = cast_column_type(df, "trace_aircraft_wd", "double")
df = cast_column_type(df, "trace_aircraft_ws", "double")
df = cast_column_type(df, "trace_aircraft_version", "double")

# Add "dbFlags" related columns
dbflags_columns_and_masks = [
    ("dbflags_military", 1),
    ("dbflags_interesting", 2),
    ("dbflags_pia", 4),
    ("dbflags_ladd", 8),
]
df = add_dbflags_columns(df, "dbFlags", dbflags_columns_and_masks)

# Add "trace_flags" related columns
trace_flags_columns_and_masks = [
    ("trace_flags_position_stale", 1),
    ("trace_flags_new_leg_start", 2),
    ("trace_flags_vertical_rate_type", 4),
    ("trace_flags_altitude_type", 8),
]
df = add_trace_flags_columns(df, "trace_flags", trace_flags_columns_and_masks)

# Add "_coordinate"
coordinate_column_name = "_coordinate"
df = add_coordinate_column(
    df,
    "trace_longitude_deg",
    "trace_latitude_deg",
    coordinate_column_name,
)

# Add "_time"
time_column_name = "_time"
df = add_time_column(df, "timestamp", "trace_relative_timestamp", time_column_name)

# Add "_date"
date_column_name = "_date"
df = add_date_column(df, time_column_name, date_column_name)

additional_options = {
    "path": delta_table_path,
    "mergeSchema": "true",
}
df.write.format("delta").options(**additional_options).partitionBy(*partitions).mode(
    "append",
).save()

job.commit()
