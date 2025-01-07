import logging
import random
import struct
import time
from pathlib import Path
from typing import Any

import polars as pl
from protos.production.iot import motor_pb2

logger = logging.getLogger(__name__)


class ProtobufWriter:
    def __init__(self, filename: Path) -> None:
        self.file = open(filename, "wb")

    def write_message(self, proto_message: motor_pb2.Motor) -> None:
        size: int = proto_message.ByteSize()
        self.file.write(struct.pack("<I", size))
        self.file.write(proto_message.SerializeToString())
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class ProtobufReader:
    def __init__(self, filename: Path) -> None:
        self.file = open(filename, "rb")

    def get_dataframe(self) -> pl.DataFrame:
        data: list[dict[str, Any]] = []
        while True:
            size_data: bytes = self.file.read(4)
            if not size_data:
                break
            size: int = struct.unpack("<I", size_data)[0]

            message_data: bytes = self.file.read(size)
            if len(message_data) != size:
                break

            motor: motor_pb2.Motor = motor_pb2.Motor()
            motor.ParseFromString(message_data)

            row: dict[str, Any] = {
                "id": motor.id,
                "timestamp_ns": motor.timestamp_ns,
                "temperature1": motor.temperature1,
                "temperature2": motor.temperature2,
                "temperature3": motor.temperature3,
                "temperature4": motor.temperature4,
                "temperature5": motor.temperature5,
            }
            data.append(row)

        df: pl.DataFrame = pl.DataFrame(data)
        return df

    def close(self) -> None:
        self.file.close()


def generate_motor_data(point_number: int) -> list[motor_pb2.Motor]:
    motors = ["motor001", "motor002", "motor003"]
    start_time_ns = time.time_ns()
    interval_ns = 1_000_000_000
    data: list[motor_pb2.Motor] = []
    for i in range(point_number):
        timestamp_ns: int = start_time_ns + (i * interval_ns)
        motor = motor_pb2.Motor()
        motor.id = random.choice(motors)
        motor.timestamp_ns = timestamp_ns
        motor.temperature1 = random.uniform(10.0, 100.0)
        motor.temperature2 = random.uniform(10.0, 100.0)
        motor.temperature3 = random.uniform(10.0, 100.0)
        motor.temperature4 = random.uniform(10.0, 100.0)
        motor.temperature5 = random.uniform(10.0, 100.0)
        data.append(motor)
    return data


def main() -> None:
    motor_data_path: Path = Path("data/motor_data.pb")
    point_number: int = 20

    # Generate data
    writer: ProtobufWriter = ProtobufWriter(motor_data_path)
    data: list[motor_pb2.Motor] = generate_motor_data(point_number)
    for motor in data:
        writer.write_message(motor)
    writer.close()

    # Read data
    reader: ProtobufReader = ProtobufReader(motor_data_path)
    df: pl.DataFrame = reader.get_dataframe()
    reader.close()
    logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
