from prefect import task
from prefect_shell import ShellOperation


@task
async def copy_to_local(
    filename: str, source_dirname: str, tmp_dirname: str
) -> list[str]:
    return await ShellOperation(
        commands=[
            f"rclone copyto {source_dirname}/{filename} {tmp_dirname}/{filename}"
        ],
    ).run()
