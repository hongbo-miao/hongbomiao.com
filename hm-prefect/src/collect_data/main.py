from prefect import flow
from prefect_shell import shell_run_command
from pydantic import BaseModel


class DataSource(BaseModel):
    source: str
    destination: str


@flow
def collect_data(data_sources: list[DataSource]) -> None:
    commands = []
    for data_source in data_sources:
        source = data_source.source
        destination = data_source.destination
        command = f"rclone copy --progress {source} {destination}"
        commands.append(command)

    shell_run_command.map(commands)


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
    collect_data(external_data_sources)
