import asyncio
import json
from pathlib import Path

from prefect import flow, task
from prefect_shell import ShellOperation
from pydantic import BaseModel


class DataSource(BaseModel):
    source_name: str
    source: str
    destination: str


@task
async def copy(source: str, destination: str) -> list[str]:
    return await ShellOperation(
        commands=[
            f"rclone lsl {source}",
            f"rclone copy --progress {source} {destination}",
        ],
    ).run()


@flow
async def collect_data(data_sources: list[DataSource]) -> None:
    tasks = []
    for data_source in data_sources:
        source_name = data_source.source_name
        source = data_source.source
        destination = data_source.destination
        copy_with_options = copy.with_options(name=f"rclone-{source_name}")
        t = asyncio.create_task(copy_with_options(source, destination))
        tasks.append(t)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    params = json.loads(Path("params.json").read_text())
    asyncio.run(
        collect_data(
            data_sources=params["data_sources"],
        )
    )
