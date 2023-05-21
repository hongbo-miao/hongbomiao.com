from prefect.client.orchestration import get_client


async def create_concurrency_limit(tag: str, concurrency_limit: int) -> None:
    async with get_client() as client:
        await client.create_concurrency_limit(
            tag=tag, concurrency_limit=concurrency_limit
        )
