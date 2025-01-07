import pathlib
import sys
import unittest

import launch
import launch_ros
import launch_testing
import pytest
import rclpy
from turtlesim.msg import Pose


@pytest.mark.rostest
def generate_test_description():
    src_path = pathlib.Path(__file__).parent.parent

    target_control_node = launch_ros.actions.Node(
        executable=sys.executable,
        arguments=[src_path.joinpath("turtle_robot/target_control_node.py").as_posix()],
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    return (
        launch.LaunchDescription(
            [
                target_control_node,
                launch_testing.actions.ReadyToTest(),
            ],
        ),
        {"target_control_node": target_control_node},
    )


class TestTargetControlNodeLink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("target_control_test_node")

    def tearDown(self):
        self.node.destroy_node()

    def test_target_control_node(self, target_control_node, proc_output):
        pose_pub = self.node.create_publisher(Pose, "turtle1/pose", 10)

        try:
            assert True
        finally:
            self.node.destroy_publisher(pose_pub)
