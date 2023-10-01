from nptdms import TdmsFile


def convert_tdms_to_parquet(data_dirname: str, tdms_filename: str) -> None:
    tdms_path = f"{data_dirname}/{tdms_filename}"
    tdms_file = TdmsFile.read(tdms_path)
    for group in tdms_file.groups():
        df = group.as_dataframe()
        print(group.name)
        print(df)
        df.to_parquet(
            f"{data_dirname}/{group.name}.parquet",
            engine="pyarrow",
            compression="zstd",
            compression_level=19,
        )
