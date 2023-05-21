import tempfile

import config
import pandas as pd
from nptdms import TdmsFile
from tasks.copy_to_local import copy_to_local
from tasks.write_to_delta_lake import write_to_delta_lake
from tasks.write_to_s3 import write_to_s3


def get_dataframe_from_tdms(tdms_path: str) -> pd.DataFrame:
    tdms_file = TdmsFile.read(tdms_path)
    assert len(tdms_file.groups()) == 1
    for group in tdms_file.groups():
        df = group.as_dataframe()
        return df


async def write_data(
    filename: str,
    source_dirname: str,
    s3_raw_path: str,
    delta_table_path: str,
    location: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dirname:
        copy_to_local_with_options = copy_to_local.with_options(
            name=f"copy-to-local-{filename}",
        )
        await copy_to_local_with_options(filename, source_dirname, tmp_dirname)
        df = get_dataframe_from_tdms(f"{tmp_dirname}/{filename}")

        write_to_delta_lake_with_options = write_to_delta_lake.with_options(
            name=f"write-to-delta-lake-{filename}",
            tags=[
                f"{config.flow_name}-write-to-delta-table-{location}-concurrency-limit"
            ],
        )
        await write_to_delta_lake_with_options(df, delta_table_path)

        write_to_s3_with_options = write_to_s3.with_options(
            name=f"write-to-s3-{filename}",
        )
        await write_to_s3_with_options(filename, source_dirname, s3_raw_path)
