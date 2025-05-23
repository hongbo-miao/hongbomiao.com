import logging
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import can
import cantools
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class TrcUtil:
    @staticmethod
    def get_dbc_schema(
        dbc_dict: dict[str, cantools.db.Database],
        unit_type: str,
    ) -> pa.Schema:
        all_fields = {
            "arbitration_id": pa.int64(),
            "channel": pa.int64(),
            "dlc": pa.int64(),
            "is_extended_id": pa.bool_(),
            "timestamp": pa.float64(),
            "_time": pa.int64(),
            "_can_id": pa.string(),
            "_can_logger_channel_id": pa.string(),
            "_unit_id": pa.string(),
        }
        for message in dbc_dict[unit_type].messages:
            for signal in message.signals:
                field_name = f"{message.name}.{signal.name}"
                if signal.choices:
                    field_type = pa.string()
                elif signal.is_float:
                    field_type = pa.float64()
                else:
                    field_type = pa.int64()
                all_fields[field_name] = field_type
        return pa.schema(all_fields)

    @staticmethod
    def initialize_schema_dict(
        dbc_dict: dict[str, cantools.db.Database],
        unit_dict: dict[str, dict[str, str]],
    ) -> dict[str, pa.Schema]:
        schema_dict = {}
        unique_types = {unit["type"] for unit in unit_dict.values()}
        for unit_type in unique_types:
            if unit_type not in schema_dict:
                schema_dict[unit_type] = TrcUtil.get_dbc_schema(dbc_dict, unit_type)
        return schema_dict

    @staticmethod
    def initialize_writer_dict(
        schema_dict: dict[str, pa.Schema],
        output_dir: Path,
        parquet_compression_method: str,
        parquet_compression_level: int,
    ) -> dict[str, pq.ParquetWriter]:
        writer_dict = {}
        for unit_type, schema in schema_dict.items():
            output_path = output_dir / f"{unit_type}.parquet"
            writer_dict[unit_type] = pq.ParquetWriter(
                output_path,
                schema,
                compression=parquet_compression_method,
                compression_level=parquet_compression_level,
            )
        return writer_dict

    @staticmethod
    def process_frame(
        frame: can.Message,
        dbc_dict: dict[str, cantools.db.Database],
        unit_dict: dict[str, dict[str, str]],
        schema_dict: dict[str, pa.Schema],
    ) -> tuple[str, str, dict[str, bool | int | float | str]]:
        unit = unit_dict[str(frame.channel)]
        unit_type = unit["type"]
        can_unit_id = unit["id"]
        dbc = dbc_dict[unit_type]
        message_definition = dbc.get_message_by_frame_id(frame.arbitration_id)
        raw_message = message_definition.decode(frame.data)
        message: dict[str, bool | int | float | str] = {}

        for signal_name, signal_value in raw_message.items():
            field_name = f"{message_definition.name}.{signal_name}"
            field_type = schema_dict[unit_type].field(field_name).type
            if pa.types.is_string(field_type):
                message[field_name] = str(signal_value)
            elif pa.types.is_integer(field_type):
                message[field_name] = int(signal_value)
            elif pa.types.is_floating(field_type):
                message[field_name] = float(signal_value)
            else:
                logger.warning(
                    f"Unexpected field type in schema: {field_type} for {field_name}",
                )
                message[field_name] = signal_value

        message.update(
            {
                "arbitration_id": frame.arbitration_id,
                "channel": frame.channel,
                "dlc": frame.dlc,
                "is_extended_id": frame.is_extended_id,
                "timestamp": frame.timestamp,
                "_time": int(Decimal(str(frame.timestamp)) * Decimal("1e9")),
                "_can_id": str(frame.arbitration_id),
                "_can_logger_channel_id": str(frame.channel),
                "_unit_id": can_unit_id,
            },
        )
        return unit_type, can_unit_id, message

    @staticmethod
    def write_batch(
        records: list[dict[str, Any]],
        unit_type: str,
        schema_dict: dict[str, pa.Schema],
        writer_dict: dict[str, pq.ParquetWriter],
    ) -> None:
        table = pa.Table.from_pylist(records, schema=schema_dict[unit_type])
        writer_dict[unit_type].write_table(table)

    @staticmethod
    def process_file(
        trc_path: Path,
        dbc_dict: dict[str, cantools.db.Database],
        unit_dict: dict[str, dict[str, str]],
        output_dir: Path,
        buffer_size: int,
        parquet_compression_method: str,
        parquet_compression_level: int,
    ) -> None:
        schema_dict = TrcUtil.initialize_schema_dict(dbc_dict, unit_dict)
        writer_dict = TrcUtil.initialize_writer_dict(
            schema_dict,
            output_dir,
            parquet_compression_method,
            parquet_compression_level,
        )

        message_count = 0
        buffer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        with can.TRCReader(trc_path) as reader:
            for frame in reader:
                unit_type, unit_id, message = TrcUtil.process_frame(
                    frame,
                    dbc_dict,
                    unit_dict,
                    schema_dict,
                )
                buffer[unit_type].append(message)

                if len(buffer[unit_type]) >= buffer_size:
                    TrcUtil.write_batch(
                        buffer[unit_type],
                        unit_type,
                        schema_dict,
                        writer_dict,
                    )
                    buffer[unit_type] = []

                if message_count % 1_000_000 == 0:
                    logger.info(f"Decoded: {message_count = }")
                message_count += 1

        # Write remaining messages in buffer
        for unit_type, messages in buffer.items():
            if messages:
                TrcUtil.write_batch(messages, unit_type, schema_dict, writer_dict)

        # Close all writers
        for writer in writer_dict.values():
            writer.close()

        logger.info(f"Decoded: 100 %, {message_count = }")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    data_dir_path = Path("data")
    trc_path = data_dir_path / Path("can.trc")
    output_dir = data_dir_path / Path("output")
    output_dir.mkdir(exist_ok=True)

    dbc_path_dict = {
        "bms": data_dir_path / Path("bms.dbc"),
        "eec": data_dir_path / Path("eec.dbc"),
    }
    unit_dict = {
        "0": {"can_logger_channel_id": "0", "type": "bms", "id": "1"},
        "1": {"can_logger_channel_id": "1", "type": "bms", "id": "2"},
        "2": {"can_logger_channel_id": "2", "type": "eec", "id": "1"},
        "3": {"can_logger_channel_id": "3", "type": "eec", "id": "2"},
    }
    buffer_size = 1_000_000
    parquet_compression_method = "zstd"
    parquet_compression_level = 19
    dbc_dict: dict[str, cantools.db.Database] = {
        unit_type: cantools.db.load_file(dbc_path)
        for unit_type, dbc_path in dbc_path_dict.items()
    }

    start_time = time.time()
    TrcUtil.process_file(
        trc_path,
        dbc_dict,
        unit_dict,
        output_dir,
        buffer_size,
        parquet_compression_method,
        parquet_compression_level,
    )
    processing_time = time.time() - start_time
    logger.info(f"Processing time: {processing_time} seconds")
