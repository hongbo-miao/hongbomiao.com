#!/usr/bin/env bash
set -e

# https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
make px4_sitl jmavsim

# Commands
pxh> commander takeoff
pxh> commander land
