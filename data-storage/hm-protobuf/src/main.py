import logging
import secrets
import struct
import time
from pathlib import Path
from typing import Any

import polars as pl
from protos.production.iot import motor_pb2

logger = logging.getLogger(__name__)


def write_protobuf_message(
    filename: Path,
    proto_message: motor_pb2.Motor,
) -> None:
    with filename.open("ab") as file:
        size: int = proto_message.ByteSize()
        file.write(struct.pack("<I", size))
        file.write(proto_message.SerializeToString())
        file.flush()


def read_protobuf_messages(filename: Path) -> pl.DataFrame:
    data: list[dict[str, Any]] = []
    with filename.open("rb") as file:
        while True:
            size_data: bytes = file.read(4)
            if not size_data:
                break
            size: int = struct.unpack("<I", size_data)[0]

            message_data: bytes = file.read(size)
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
    return pl.DataFrame(data)


def generate_motor_data(point_number: int) -> list[motor_pb2.Motor]:
    motors = ["motor001", "motor002", "motor003"]
    start_time_ns = time.time_ns()
    interval_ns = 1_000_000_000
    data: list[motor_pb2.Motor] = []
    for i in range(point_number):
        timestamp_ns: int = start_time_ns + (i * interval_ns)
        motor = motor_pb2.Motor()
        system_random = secrets.SystemRandom()
        motor.id = system_random.choice(motors)
        motor.timestamp_ns = timestamp_ns
        motor.temperature1 = system_random.uniform(10.0, 100.0)
        motor.temperature2 = system_random.uniform(10.0, 100.0)
        motor.temperature3 = system_random.uniform(10.0, 100.0)
        motor.temperature4 = system_random.uniform(10.0, 100.0)
        motor.temperature5 = system_random.uniform(10.0, 100.0)
        data.append(motor)
    return data


def main() -> None:
    # Generate data
    motor_data_path: Path = Path("data/motor_data.pb")
    point_number: int = 20

    motor_data: list[motor_pb2.Motor] = generate_motor_data(point_number)
    for motor in motor_data:
        write_protobuf_message(motor_data_path, motor)

    # Read data
    df = read_protobuf_messages(motor_data_path)
    logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
