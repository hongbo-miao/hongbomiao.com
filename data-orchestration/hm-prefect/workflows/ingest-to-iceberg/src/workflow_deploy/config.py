from typing import Any

BASE_WORKFLOW_NAME = "ingest-to-iceberg"
DEVELOPMENT_DEPLOYMENT: dict[str, Any] = {
    "schedule": None,
    "parameters": {
        "config": {
            "spark_connect_url": "sc://spark-connect-svc.development-hm-spark.svc:15002",
            "parquet_data_path": "/data",
        },
    },
    "docker_image_tag": "development",
    "work_pool_name": "hm-work-pool",
}
PRODUCTION_DEPLOYMENT: dict[str, Any] = {
    "schedule": None,
    "parameters": {
        "config": {
            "spark_connect_url": "sc://spark-connect-svc.production-hm-spark.svc:15002",
            "parquet_data_path": "/data",
        },
    },
    "docker_image_tag": "latest",
    "work_pool_name": "hm-work-pool",
}
