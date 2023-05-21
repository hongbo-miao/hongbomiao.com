import asyncio

import config
from utils.create_aws_credentials_block import create_aws_credentials_block
from utils.create_concurrency_limit import create_concurrency_limit
from utils.create_kubernetes_job_block import create_kubernetes_job_block


async def main() -> None:
    await create_concurrency_limit(
        f"{config.flow_name}-write-to-delta-table-motor-concurrency-limit", 1
    )
    await create_aws_credentials_block(
        config.flow_name,
        config.aws_default_region,
        config.aws_access_key_id,
        config.aws_secret_access_key,
    )
    await create_kubernetes_job_block(config.flow_name)


if __name__ == "__main__":
    asyncio.run(main())
