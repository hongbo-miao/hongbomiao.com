ros-run-cpp-talker:
	ros2 run demo_nodes_cpp talker
ros-run-cpp-talker:
	ros2 run demo_nodes_cpp listener

ros-run-python-talker:
	ros2 run demo_nodes_py talker
ros-run-python-listener:
	ros2 run demo_nodes_py listener

ros2-pkg-create-cpp-package:
	cd src && ros2 pkg create --build-type=ament_cmake --node-name=hm_cpp_node --dependencies=rclcpp hm_cpp_package
ros2-pkg-create-python-package:
	cd src && ros2 pkg create --build-type=ament_python --node-name=hm_python_node --dependencies=rclpy hm_python_package
colcon-build:
	colcon build
colcon-build-package:
	colcon build --packages-select=hm_cpp_package
	colcon build --packages-select=hm_python_package
colcon-test:
	colcon test
colcon-test-package:
	colcon test --packages-select=hm_cpp_package
	colcon test --packages-select=hm_python_package

ros2-init:
	source install/local_setup.bash
ros2-run-hm-python-package:
	colcon build --packages-select=hm_python_package && ros2 run hm_python_package hm_python_node
ros2-run-hm-cpp-package:
	colcon build --packages-select=hm_cpp_package && ros2 run hm_cpp_package hm_cpp_node

poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev
poetry-run-dev:
	poetry run poe dev
