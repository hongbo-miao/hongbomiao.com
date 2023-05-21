import asyncio

from dotenv import dotenv_values
from utils.create_aws_credentials_block import create_aws_credentials_block
from utils.create_concurrency_limit import create_concurrency_limit
from utils.create_kubernetes_job_block import create_kubernetes_job_block


async def set_up(
    flow_name: str,
    aws_default_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
) -> None:
    await create_concurrency_limit(
        f"{flow_name}-write-to-delta-table-motor-concurrency-limit", 1
    )
    await create_aws_credentials_block(
        flow_name, aws_default_region, aws_access_key_id, aws_secret_access_key
    )
    await create_kubernetes_job_block(flow_name)


if __name__ == "__main__":
    config = dotenv_values("tools/set_up/.env.production.local")

    external_flow_name = config["FLOW_NAME"]
    external_aws_default_region = config["AWS_DEFAULT_REGION"]
    external_aws_access_key_id = config["AWS_ACCESS_KEY_ID"]
    external_aws_secret_access_key = config["AWS_SECRET_ACCESS_KEY"]

    asyncio.run(
        set_up(
            external_flow_name,
            external_aws_default_region,
            external_aws_access_key_id,
            external_aws_secret_access_key,
        )
    )
