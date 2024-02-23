import time
from decimal import Decimal
from pathlib import Path

import can
import cantools
import pandas as pd


def load_dbc_dict(dbc_path_dict: dict[str, Path]) -> dict[str, cantools.db.Database]:
    return {
        type: cantools.db.load_file(dbc_path)
        for type, dbc_path in dbc_path_dict.items()
    }


def decode_can_data(
    blf_path: Path,
    dbc_dict: dict[str, cantools.db.Database],
    unit_dict: dict[str, dict[str, str]],
) -> dict[str, list[dict[str, bool | float | int | str]]]:
    blf_size_bytes = blf_path.stat().st_size
    unit_type_and_unit_id_dict: dict[str, list[dict[str, str | int | float]]] = {}
    message_count = 0
    with can.BLFReader(blf_path) as blf_reader:
        for frame in blf_reader:
            current_bytes = blf_reader.file.tell()
            unit = unit_dict[str(frame.channel)]
            message_definition = dbc_dict[unit["type"]].get_message_by_frame_id(
                frame.arbitration_id
            )
            message = message_definition.decode(frame.data)
            # Convert cantools.database.can.signal.NamedSignalValue to str
            message = {
                signal_name: (
                    str(signal_value)
                    if isinstance(
                        signal_value,
                        cantools.database.can.signal.NamedSignalValue,
                    )
                    else signal_value
                )
                for signal_name, signal_value in message.items()
            }
            # Add message_definition.name as prefix
            message = {
                f"{message_definition.name}.{signal_name}": signal_value
                for signal_name, signal_value in message.items()
            }
            message.update(
                {
                    "arbitration_id": frame.arbitration_id,
                    "channel": frame.channel,
                    "dlc": frame.dlc,
                    "is_extended_id": frame.is_extended_id,
                    "timestamp": frame.timestamp,
                    # Avoid `int(frame.timestamp * 1e9` as it leads to time drift due to floating-point arithmetic
                    "_time": int(Decimal(str(frame.timestamp)) * Decimal("1e9")),
                    "_can_id": str(frame.arbitration_id),
                    "_can_logger_channel_id": str(frame.channel),
                    "_unit_id": unit["id"],
                }
            )
            unit_type_and_unit_id = f"{unit['type']}_{unit['id']}"
            unit_type_and_unit_id_dict.setdefault(unit_type_and_unit_id, []).append(
                message
            )
            if message_count % 1000000 == 0:
                print(
                    f"Decoded: {round(current_bytes * 100.0 / blf_size_bytes)} %, {message_count = }"
                )
            message_count += 1
    print(f"Decoded: 100 %, {message_count = }")
    return unit_type_and_unit_id_dict


if __name__ == "__main__":
    start_time = time.time()
    data_dir_path = Path("data")
    blf_path = data_dir_path / Path("can.blf")
    dbc_path_dict = {
        "bms": data_dir_path / Path("bms.dbc"),
        "eec": data_dir_path / Path("eec.dbc"),
    }
    unit_dict = {
        "0": {"can_logger_channel_id": "0", "type": "bms", "id": "1"},
        "1": {"can_logger_channel_id": "1", "type": "bms", "id": "2"},
        "2": {"can_logger_channel_id": "2", "type": "eec", "id": "1"},
        "3": {"can_logger_channel_id": "3", "type": "eec", "id": "2"},
        "4": {"can_logger_channel_id": "4", "type": "eec", "id": "3"},
        "5": {"can_logger_channel_id": "5", "type": "eec", "id": "4"},
    }

    dbc_dict = load_dbc_dict(dbc_path_dict)
    unit_type_and_unit_id_dict = decode_can_data(blf_path, dbc_dict, unit_dict)
    print(f"Deserializing time: {time.time() - start_time} seconds")

    for unit_type_and_unit_id, data in unit_type_and_unit_id_dict.items():
        df = pd.DataFrame(data)
        df.to_parquet(f"{unit_type_and_unit_id}.parquet", engine="pyarrow")
