import asyncio

import config
from utils.create_kubernetes_job_block import create_kubernetes_job_block


async def main() -> None:
    await create_kubernetes_job_block(config.flow_name)


if __name__ == "__main__":
    asyncio.run(main())
