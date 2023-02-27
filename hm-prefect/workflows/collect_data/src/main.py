import asyncio
import json
from pathlib import Path

from prefect import flow
from prefect_shell import shell_run_command
from pydantic import BaseModel


class DataSource(BaseModel):
    sourceName: str
    source: str
    destination: str


@flow
async def collect_data(data_sources: list[DataSource]) -> None:
    tasks = []
    for data_source in data_sources:
        source_name = data_source.sourceName
        source = data_source.source
        destination = data_source.destination
        copy = shell_run_command.with_options(name=f"rclone-{source_name}")
        task = asyncio.create_task(
            copy(
                command=f"rclone copy --progress {source} {destination}",
                helper_command=f"rclone lsl {source}",
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    params = json.loads(Path("params.json").read_text())
    asyncio.run(
        collect_data(
            data_sources=params["dataSources"],
        )
    )
