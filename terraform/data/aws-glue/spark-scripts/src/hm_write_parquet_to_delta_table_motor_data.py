import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

raw_parquet_path = "s3://hongbomiao-bucket/data/parquet/motor/"
delta_table_path = "s3://hongbomiao-bucket/data/delta-tables/motor_data/"
partition_list = ["_event_id"]

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

additional_options = {
    "path": delta_table_path,
    "mergeSchema": "true",
}
df = s3_node.toDF()
df.write.format("delta").options(**additional_options).partitionBy(
    *partition_list
).mode("overwrite").save()

job.commit()
