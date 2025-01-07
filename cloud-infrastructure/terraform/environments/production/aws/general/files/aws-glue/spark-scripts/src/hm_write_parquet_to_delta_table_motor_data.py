import logging
import os
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

raw_parquet_paths = ["s3://hm-production-bucket/data/parquet/motor/"]
delta_table_path = "s3://hm-production-bucket/data/delta-tables/motor_data/"
partitions = ["_event_id"]

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

additional_options = {
    "path": delta_table_path,
    "mergeSchema": "true",
}
df.write.format("delta").options(**additional_options).partitionBy(*partitions).mode(
    "overwrite",
).save()

job.commit()
