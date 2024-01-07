import asyncio
import json
from pathlib import Path

import config
from pydantic import BaseModel
from utils.create_aws_credentials_block import create_aws_credentials_block
from utils.create_concurrency_limit import create_concurrency_limit
from utils.create_kubernetes_job_block import create_kubernetes_job_block


class DataSource(BaseModel):
    source_dirname: str
    location: str


async def main(data_sources: list[DataSource]) -> None:
    create_concurrency_limit_tasks = []
    for data_source in data_sources:
        location = data_source["location"]
        t = asyncio.create_task(
            create_concurrency_limit(
                config.flow_name,
                f"write-to-delta-table-{location}",
                1,
            )
        )
        create_concurrency_limit_tasks.append(t)
    await asyncio.gather(*create_concurrency_limit_tasks)
    await create_aws_credentials_block(
        config.flow_name,
        config.aws_default_region,
        config.aws_access_key_id,
        config.aws_secret_access_key,
    )
    await create_kubernetes_job_block(config.flow_name)


if __name__ == "__main__":
    params = json.loads(Path("params.json").read_text())
    asyncio.run(main(data_sources=params["data_sources"]))
