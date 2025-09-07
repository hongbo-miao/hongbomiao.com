#!/usr/bin/env bash
set -e

echo "# Install ROS"
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
locale
sudo apt-get install --yes locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale

sudo apt-get install --yes software-properties-common
sudo add-apt-repository universe
sudo apt-get install --yes curl
sudo curl --silent --fail --show-error --location https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get install --yes ros-humble-desktop

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "=================================================="

echo "# Install colcon"
# https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html
sudo apt-get install --yes python3-colcon-common-extensions
echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc
echo "=================================================="

echo "# Install turtlesim"
sudo apt-get install --yes ros-humble-turtlesim
echo "=================================================="

echo "# Set up"
echo "source /media/psf/Home/Clouds/Git/hongbomiao.com/robotics/robot-operating-system/install/setup.bash" >> ~/.bashrc
# source ~/.bashrc
echo "=================================================="
