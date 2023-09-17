import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

raw_parquet_path = "s3://hongbomiao-bucket/data/parquet/motor/"
delta_table_path = "s3://hongbomiao-bucket/data/delta-tables/motor/"
partition_list = ["_event_id"]

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node S3 bucket
S3bucket_node1 = glueContext.create_dynamic_frame.from_options(
    format_options={},
    connection_type="s3",
    format="parquet",
    connection_options={
        "paths": [raw_parquet_path],
        "recurse": True,
    },
    transformation_ctx="S3bucket_node1",
)

# Script generated for node sink_to_delta_lake
additional_options = {
    "path": delta_table_path,
    "write.parquet.compression-codec": "snappy",
    "mergeSchema": "true",
}
sink_to_delta_lake_node3_df = S3bucket_node1.toDF()
sink_to_delta_lake_node3_df.write.format("delta").options(
    **additional_options
).partitionBy(*partition_list).mode("overwrite").save()

job.commit()
