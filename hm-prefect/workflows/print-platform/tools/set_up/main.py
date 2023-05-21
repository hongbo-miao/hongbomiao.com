import asyncio

from utils.create_kubernetes_job_block import create_kubernetes_job_block


async def main(
    flow_name: str,
) -> None:
    await create_kubernetes_job_block(flow_name)


if __name__ == "__main__":
    external_flow_name = "print-platform"
    asyncio.run(
        main(
            external_flow_name,
        )
    )
