from typing import Any

from prefect import Flow
from prefect.docker import DockerImage
from workflow_deploy import config


async def create_deployment(
    environment: str,
    workflow: Flow,
    docker_image_name: str,
    deployment: dict[str, Any],
) -> None:
    deployment_name = f"hm-{environment}-{config.BASE_WORKFLOW_NAME}"
    await workflow.deploy(
        name=deployment_name,
        work_pool_name=deployment["work_pool_name"],
        parameters=deployment["parameters"],
        schedule=deployment["schedule"],
        image=DockerImage(
            name=docker_image_name,
            tag=deployment["docker_image_tag"],
            dockerfile="Dockerfile",
        ),
        push=False,
    )
