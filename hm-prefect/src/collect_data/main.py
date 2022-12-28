import asyncio

from prefect import flow
from prefect_shell import shell_run_command
from pydantic import BaseModel


class DataSource(BaseModel):
    source: str
    destination: str


@flow
async def collect_data(data_sources: list[DataSource]) -> None:
    await asyncio.gather(
        *[
            shell_run_command(
                command=f"rclone copy --progress {data_source.source} {data_source.destination}",
                helper_command=f"rclone lsl {data_source.source}",
            )
            for data_source in data_sources
        ]
    )


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
