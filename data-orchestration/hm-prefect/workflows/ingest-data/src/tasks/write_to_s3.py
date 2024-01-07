from prefect import task
from prefect_shell import ShellOperation


@task
async def write_to_s3(filename: str, source_dirname: str, s3_path: str) -> list[str]:
    return await ShellOperation(
        commands=[
            f"rclone copyto --s3-no-check-bucket {source_dirname}/{filename} {s3_path}/{filename}"
        ],
    ).run()
