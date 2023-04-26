import time

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter


def generate_iot_tdms(
    data_dirname: str, tdms_filename: str, data_point_count: int
) -> None:
    tdms_path = f"{data_dirname}/{tdms_filename}"

    root_object = RootObject(
        properties={
            "author": "Hongbo Miao",
            "description": "IoT data",
        }
    )

    motor_group_name = "motor"
    motor_group = GroupObject(
        motor_group_name,
        properties={
            "motor_model": "XYZ-123",
            "serial_number": "SN-456",
        },
    )

    current_data = np.random.rand(data_point_count) * 10
    voltage_data = np.random.rand(data_point_count) * 20
    temperature_data = np.random.rand(data_point_count) * 50 + 25
    timestamp_data = np.array([time.time() + i for i in range(data_point_count)])

    current_channel = ChannelObject(
        motor_group_name, "current", current_data, properties={"units": "A"}
    )
    voltage_channel = ChannelObject(
        motor_group_name, "voltage", voltage_data, properties={"units": "V"}
    )
    temperature_channel = ChannelObject(
        motor_group_name, "temperature", temperature_data, properties={"units": "C"}
    )
    timestamp_channel = ChannelObject(
        motor_group_name,
        "timestamp",
        timestamp_data,
        properties={"units": "s"},
    )

    with TdmsWriter(tdms_path) as tdms_writer:
        tdms_writer.write_segment(
            [
                root_object,
                motor_group,
                current_channel,
                voltage_channel,
                temperature_channel,
                timestamp_channel,
            ]
        )
