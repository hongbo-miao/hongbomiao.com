from prefect import get_client


async def create_concurrency_limit(
    flow_name: str, task_name: str, concurrency_limit: int
) -> None:
    async with get_client() as client:
        await client.create_concurrency_limit(
            tag=f"{flow_name}-{task_name}-concurrency-limit",
            concurrency_limit=concurrency_limit,
        )
