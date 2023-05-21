import asyncio
import json
from pathlib import Path

from prefect import flow
from pydantic import BaseModel
from tasks.get_missing_files import get_missing_files
from utils.write_data import write_data


class DataSource(BaseModel):
    source_dirname: str
    s3_path: str
    delta_table_path: str


@flow
async def ingest_data(data_sources: list[DataSource]) -> None:
    get_missing_files_tasks = []
    for data_source in data_sources:
        source_dirname = data_source.source_dirname
        s3_path = data_source.s3_path
        delta_table_path = data_source.delta_table_path
        get_missing_files_with_options = get_missing_files.with_options(
            name=f"get-missing-files-{source_dirname}"
        )
        t = asyncio.create_task(
            get_missing_files_with_options(source_dirname, s3_path, delta_table_path)
        )
        get_missing_files_tasks.append(t)
    missing_files_list = await asyncio.gather(*get_missing_files_tasks)

    write_tasks = []
    for missing_files in missing_files_list:
        for filename, source_dirname, s3_path, delta_table_path in missing_files:
            t = asyncio.create_task(
                write_data(filename, source_dirname, s3_path, delta_table_path)
            )
            write_tasks.append(t)
    await asyncio.gather(*write_tasks)


if __name__ == "__main__":
    params = json.loads(Path("params.json").read_text())
    asyncio.run(ingest_data(data_sources=params["data_sources"]))
