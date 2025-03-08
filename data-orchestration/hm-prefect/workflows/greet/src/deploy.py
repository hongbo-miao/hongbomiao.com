import asyncio
import logging

from main import greet
from workflow_deploy import config
from workflow_deploy.args import get_args
from workflow_deploy.environments import Environments
from workflow_deploy.utils.create_deployment import create_deployment


async def hm_greet() -> None:
    args = get_args()

    match args.environment:
        case Environments.development.value:
            deployment = config.DEVELOPMENT_DEPLOYMENT
        case Environments.production.value:
            deployment = config.PRODUCTION_DEPLOYMENT
        case _:
            logging.error("Not supported environment.")
            return

    docker_image_name = f"harbor.hongbomiao.com/hm/prefect-{config.BASE_WORKFLOW_NAME}"
    await create_deployment(
        args.environment,
        greet,
        docker_image_name,
        deployment,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(hm_greet())
