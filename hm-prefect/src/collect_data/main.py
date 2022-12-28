import asyncio

from prefect import flow
from prefect_shell import shell_run_command
from pydantic import BaseModel


class DataSource(BaseModel):
    source: str
    destination: str


@flow
async def collect_data(data_sources: list[DataSource]) -> None:
    tasks = []
    for data_source in data_sources:
        src = data_source.source
        dest = data_source.destination
        copy = shell_run_command.with_options(name=f"copy | {src}")
        task = asyncio.create_task(
            copy(
                command=f"rclone copy --progress {src} {dest}",
                helper_command=f"rclone lsl {src}",
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    external_data_sources = [
        {
            "source": "hm-ubuntu:/tmp/rclone-backup/data1",
            "destination": "/tmp/rclone-backup/data1",
        },
        {
            "source": "hm-ubuntu:/tmp/rclone-backup/data2",
            "destination": "/tmp/rclone-backup/data2",
        },
    ]
    asyncio.run(collect_data(external_data_sources))
