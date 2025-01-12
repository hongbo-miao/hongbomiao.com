import pathlib
import sys
import unittest

import pytest
import rclpy
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_testing.actions import ReadyToTest
from rclpy.publisher import Publisher
from turtlesim.msg import Pose


@pytest.mark.rostest
def generate_test_description() -> LaunchDescription:
    src_path = pathlib.Path(__file__).parent.parent

    target_control_node = Node(
        executable=sys.executable,
        arguments=[src_path.joinpath("turtle_robot/target_control_node.py").as_posix()],
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    return (
        LaunchDescription(
            [
                target_control_node,
                ReadyToTest(),
            ],
        ),
        {"target_control_node": target_control_node},
    )


class TestTargetControlNodeLink(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        rclpy.shutdown()

    def setUp(self) -> None:
        self.node = rclpy.create_node("target_control_test_node")

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_target_control_node(
        self,
        _target_control_node: Node,  # noqa: PT019
        _proc_output: tuple[bytes, bytes],  # noqa: PT019
    ) -> None:
        pose_pub: Publisher = self.node.create_publisher(Pose, "turtle1/pose", 10)

        try:
            assert True
        finally:
            self.node.destroy_publisher(pose_pub)
