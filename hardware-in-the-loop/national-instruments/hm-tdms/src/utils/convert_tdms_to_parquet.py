import logging

import polars as pl
from nptdms import TdmsFile

logger = logging.getLogger(__name__)


def convert_tdms_to_parquet(data_dirname: str, tdms_filename: str) -> None:
    tdms_path = f"{data_dirname}/{tdms_filename}"
    tdms_file = TdmsFile.read(tdms_path)
    for group in tdms_file.groups():
        df = pl.from_pandas(group.as_dataframe())
        logger.info(group.name)
        logger.info(df)
        df.write_parquet(
            f"{data_dirname}/{group.name}.parquet",
            compression="zstd",
            compression_level=19,
        )
