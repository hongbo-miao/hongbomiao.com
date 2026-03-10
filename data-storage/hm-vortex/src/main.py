import logging
from pathlib import Path

import polars as pl
import pyarrow as pa
import vortex
import vortex.expr as ve
import vortex.io

logger = logging.getLogger(__name__)


def convert_struct_array_to_polars_dataframe(
    struct_array: vortex.Array,
) -> pl.DataFrame:
    arrow_struct = struct_array.to_arrow_array()
    arrow_table = pa.Table.from_arrays(
        arrow_struct.flatten(),
        names=[field.name for field in arrow_struct.type],
    )
    return pl.from_arrow(arrow_table)


def main() -> None:
    file_path = Path("data/sensors.vortex")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a PyArrow table representing sensor readings
    sensor_table = pa.table(
        {
            "sensor_id": [
                "sensor_a",
                "sensor_b",
                "sensor_a",
                "sensor_c",
                "sensor_b",
                "sensor_a",
            ],
            "temperature": [22.5, 18.3, 23.1, None, 19.0, 22.8],
            "humidity": [45.0, 60.2, 44.5, 55.0, None, 46.1],
            "is_anomaly": [False, False, False, True, False, True],
        },
    )
    logger.info(f"Created sensor table with {sensor_table.num_rows} rows")

    # Convert to a Vortex array
    sensor_array = vortex.array(sensor_table)
    logger.info(f"Vortex array dtype: {sensor_array.dtype}")

    # Compress the array using Vortex's automatic compression
    compressed_array = vortex.compress(sensor_array)
    logger.info(f"Compressed array: {compressed_array}")

    # Write to a Vortex file
    vortex.io.write(compressed_array, str(file_path))
    logger.info(f"Written Vortex file to {file_path}")

    # Read back from the Vortex file
    vortex_file = vortex.open(str(file_path))
    logger.info(
        f"Opened Vortex file: dtype={vortex_file.dtype}, row_count={len(vortex_file)}",
    )

    # Scan all rows
    all_rows = vortex_file.scan().read_all()
    all_rows_dataframe = convert_struct_array_to_polars_dataframe(all_rows)
    logger.info(f"All rows:\n{all_rows_dataframe}")

    # Scan with column projection and row filter:
    # Select only sensor_id and temperature where temperature > 20
    high_temperature_rows = vortex_file.scan(
        ["sensor_id", "temperature"],
        expr=ve.column("temperature") > ve.literal(vortex.float_(64), 20.0),
    ).read_all()
    high_temperature_dataframe = convert_struct_array_to_polars_dataframe(
        high_temperature_rows,
    )
    logger.info(f"High temperature readings (> 20.0):\n{high_temperature_dataframe}")

    # Scan with limit
    limited_rows = vortex_file.scan(limit=2).read_all()
    limited_dataframe = convert_struct_array_to_polars_dataframe(limited_rows)
    logger.info(f"First 2 rows:\n{limited_dataframe}")

    # Scan anomaly rows
    anomaly_rows = vortex_file.scan(
        ["sensor_id", "temperature", "humidity"],
        expr=ve.column("is_anomaly") == ve.literal(vortex.bool_(), value=True),
    ).read_all()
    anomaly_dataframe = convert_struct_array_to_polars_dataframe(anomaly_rows)
    logger.info(f"Anomaly readings:\n{anomaly_dataframe}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
