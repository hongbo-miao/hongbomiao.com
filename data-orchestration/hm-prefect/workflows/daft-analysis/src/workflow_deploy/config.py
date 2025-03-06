from typing import Any

BASE_WORKFLOW_NAME = "daft-analysis"
DEVELOPMENT_DEPLOYMENT: dict[str, Any] = {
    "schedule": None,
    "parameters": {},
    "docker_image_tag": "development",
    "work_pool_name": "hm-work-pool",
}
PRODUCTION_DEPLOYMENT: dict[str, Any] = {
    "schedule": None,
    "parameters": {},
    "docker_image_tag": "latest",
    "work_pool_name": "hm-work-pool",
}
