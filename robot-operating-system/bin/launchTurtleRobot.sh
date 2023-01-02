#!/usr/bin/env bash
set -e

colcon build --packages-select=turtle_robot
colcon build --packages-select=turtle_robot_launch
ros2 launch turtle_robot_launch turtle_robot.launch.py
