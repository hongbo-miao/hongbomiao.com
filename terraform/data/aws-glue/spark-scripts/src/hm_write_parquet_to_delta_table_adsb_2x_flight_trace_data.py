import sys

from awsglue import DynamicFrame
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

raw_parquet_path = "s3://hongbomiao-bucket/data/raw-parquet/adsb_2x_flight_trace_data/"
delta_table_path = "s3://hongbomiao-bucket/data/delta-tables/adsb_2x_flight_trace_data/"
partition_list = ["_date"]


def spark_sql_query(
    glue_context: GlueContext, query: str, mapping: dict, transformation_ctx: str
) -> DynamicFrame:
    for alias, frame in mapping.items():
        frame.toDF().createOrReplaceTempView(alias)
    res = spark.sql(query)
    return DynamicFrame.fromDF(res, glue_context, transformation_ctx)


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
        "paths": [raw_parquet_path],
        "recurse": True,
    },
    transformation_ctx="s3_node",
)

sql_query = """
    select
        *,
        case when (trace_db_flags & 1) > 0 then true else false end as trace_position_stale,
        case when (trace_db_flags & 2) > 0 then true else false end as trace_new_leg_start,
        case when (trace_db_flags & 4) > 0 then 'geometric' else 'barometric' end as trace_vertical_rate_type,
        case when (trace_db_flags & 8) > 0 then 'geometric' else 'barometric' end as trace_altitude_type,
        concat('POINT (', trace_longitude_deg, ' ', trace_latitude_deg, ')') as _coordinate
    from my_table;
"""
sql_query_node = spark_sql_query(
    glue_context,
    query=sql_query,
    mapping={"my_table": s3_node},
    transformation_ctx="sql_query_node",
)

additional_options = {
    "path": delta_table_path,
    "mergeSchema": "true",
}
df = sql_query_node.toDF()
df.write.format("delta").options(**additional_options).partitionBy(
    *partition_list
).mode("overwrite").save()

job.commit()
