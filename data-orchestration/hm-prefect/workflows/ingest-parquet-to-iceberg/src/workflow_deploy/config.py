from typing import Any

from prefect.client.schemas.schedules import CronSchedule

BASE_WORKFLOW_NAME = "ingest-parquet-to-iceberg"
DEVELOPMENT_DEPLOYMENT: dict[str, Any] = {
    "schedule": None,
    "parameters": {
        "config": {
            "spark_connect_url": "sc://spark-connect-large.hongbomiao.com:443/;use_ssl=true",
            "checkpoint_base_path": "s3a://development-hm-spark-checkpoints",
            "source_path": "s3a://development-hm-data-raw/motor/",
            "catalog": "development",
            "namespace": "motor_db",
            "table_name": "motor_data",
            "partition_column": "_time",
        },
    },
    "docker_image_tag": "latest",
    "work_pool_name": "hm-work-pool",
}
STAGING_DEPLOYMENT: dict[str, Any] = {
    "schedule": None,
    "parameters": {
        "config": {
            "spark_connect_url": "sc://spark-connect-large.hongbomiao.com:443/;use_ssl=true",
            "checkpoint_base_path": "s3a://staging-hm-spark-checkpoints",
            "source_path": "s3a://staging-hm-data-raw/motor/",
            "catalog": "staging",
            "namespace": "motor_db",
            "table_name": "motor_data",
            "partition_column": "_time",
        },
    },
    "docker_image_tag": "latest",
    "work_pool_name": "hm-work-pool",
}
PRODUCTION_DEPLOYMENT: dict[str, Any] = {
    "schedule": CronSchedule(cron="5 * * * *", timezone="UTC"),
    "parameters": {
        "config": {
            "spark_connect_url": "sc://spark-connect-large.hongbomiao.com:443/;use_ssl=true",
            "checkpoint_base_path": "s3a://production-hm-spark-checkpoints",
            "source_path": "s3a://production-hm-data-raw/motor/",
            "catalog": "production",
            "namespace": "motor_db",
            "table_name": "motor_data",
            "partition_column": "_time",
        },
    },
    "docker_image_tag": "latest",
    "work_pool_name": "hm-work-pool",
}
