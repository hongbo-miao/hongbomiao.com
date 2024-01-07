import json

from prefect import task
from prefect_shell import ShellOperation


def extract_missing_files(log: list[str]) -> list[str]:
    missing_files = []
    for line in log:
        data = json.loads(line)
        if "object" in data:
            missing_files.append(data["object"])
    return missing_files


@task
async def get_missing_files(
    source_dirname: str, s3_raw_path: str, delta_table_path: str, location: str
) -> list[tuple[str, str, str, str, str]]:
    log = await ShellOperation(
        commands=[
            f'rclone copy --dry-run --include="*.tdms" --use-json-log {source_dirname} {s3_raw_path}',
        ],
        stream_output=False,
    ).run()
    missing_files = extract_missing_files(log)
    return [
        (filename, source_dirname, s3_raw_path, delta_table_path, location)
        for filename in missing_files
    ]
