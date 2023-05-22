from utils.convert_tdms_to_parquet import convert_tdms_to_parquet
from utils.generate_iot_tdms import generate_iot_tdms


def main(data_dirname: str, tdms_filename: str, row_count: int) -> None:
    generate_iot_tdms(data_dirname, tdms_filename, row_count)
    convert_tdms_to_parquet(data_dirname, tdms_filename)


if __name__ == "__main__":
    external_data_dirname = "data"
    external_tdms_filename = "iot.tdms"
    external_row_count = 1000000
    main(external_data_dirname, external_tdms_filename, external_row_count)
