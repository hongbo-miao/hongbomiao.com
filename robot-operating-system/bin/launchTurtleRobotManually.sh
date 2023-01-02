#!/usr/bin/env bash
set -e

# Setup
source install/local_setup.bash

# Start the turtlesim
ros2 run turtlesim turtlesim_node

# Add turtle_robot
ros2 service call /spawn turtlesim/srv/Spawn "{x: 2.0, y: 2.0, theta: 0.0, name: 'turtle_robot'}"

# Build the turtle_robot
colcon build --packages-select=turtle_robot --symlink-install
ros2 run turtle_robot target_control_node
ros2 run turtle_robot turtle_robot_control_node

# Control the target turtle
ros2 run turtlesim turtle_teleop_key
