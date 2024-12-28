import polars as pl
from nptdms import TdmsFile


def convert_tdms_to_parquet(data_dirname: str, tdms_filename: str) -> None:
    tdms_path = f"{data_dirname}/{tdms_filename}"
    tdms_file = TdmsFile.read(tdms_path)
    for group in tdms_file.groups():
        df = pl.from_pandas(group.as_dataframe())
        print(group.name)
        print(df)
        df.write_parquet(
            f"{data_dirname}/{group.name}.parquet",
            compression="zstd",
            compression_level=19,
        )
