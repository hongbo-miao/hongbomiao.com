#!/usr/bin/env bash
set -e

# Start the turtlesim
ros2 run turtlesim turtlesim_node

# Add robot_turtle
ros2 service call /spawn turtlesim/srv/Spawn "{x: 2.0, y: 2.0, theta: 0.0, name: 'robot_turtle'}"

# Build the robot_turtle
colcon build --packages-select=hm_turtle --symlink-install
ros2 run hm_turtle turtle_controller

# Control the target turtle
ros2 run turtlesim turtle_teleop_key
