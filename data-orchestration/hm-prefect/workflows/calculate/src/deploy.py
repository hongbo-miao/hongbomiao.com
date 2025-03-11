import asyncio
import logging

from main import calculate
from workflow_deploy import config
from workflow_deploy.args import get_args
from workflow_deploy.environments import Environments
from workflow_deploy.utils.create_deployment import create_deployment


async def hm_calculate() -> None:
    args = get_args()

    match args.environment:
        case Environments.development.value:
            deployment = config.DEVELOPMENT_DEPLOYMENT
        case Environments.production.value:
            deployment = config.PRODUCTION_DEPLOYMENT
        case _:
            logging.error(f"Not supported environment: {args.environment}")
            return

    docker_image_name = f"harbor.hongbomiao.com/hm/prefect-{config.BASE_WORKFLOW_NAME}"
    await create_deployment(
        args.environment,
        calculate,
        docker_image_name,
        deployment,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(hm_calculate())
