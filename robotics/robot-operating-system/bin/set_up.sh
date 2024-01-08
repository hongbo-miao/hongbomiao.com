#!/usr/bin/env bash
set -e

echo "# Install pyenv"
# https://github.com/pyenv/pyenv-installer
sudo apt install --yes git
curl https://pyenv.run | bash
{
  export PYENV_ROOT="$HOME/.pyenv"
  command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
} >> ~/.bashrc
pyenv install 3.11
echo "=================================================="

echo "# Install Poetry"
# https://python-poetry.org/docs/
curl --silent --fail --show-error --location https://install.python-poetry.org | python3 -
# shellcheck disable=SC2016
echo 'export PATH="/home/parallels/.local/bin:$PATH"' >> ~/.bashrc
echo "=================================================="

echo "# Install ROS"
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
locale
sudo apt update
sudo apt install --yes locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale

sudo apt install --yes software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl --silent --fail --show-error --location https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade
sudo apt install --yes ros-humble-desktop

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "=================================================="

echo "# Install colcon"
# https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html
sudo apt install --yes python3-colcon-common-extensions
echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc
echo "=================================================="

echo "# Install turtlesim"
sudo apt install --yes ros-humble-turtlesim
echo "=================================================="

echo "# Set up"
echo "source /media/psf/Home/Clouds/Git/hongbomiao.com/robotics/robot-operating-system/install/setup.bash" >> ~/.bashrc
# source ~/.bashrc
echo "=================================================="
