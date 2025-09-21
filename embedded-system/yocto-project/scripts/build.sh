#!/usr/bin/env bash
set -e

# https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html

echo "# Clone Poky"
sudo apt-get install --yes git
git clone git://git.yoctoproject.org/poky
cd poky
git checkout kirkstone
source oe-init-build-env

echo "# Build"
# https://wiki.koansoftware.com/index.php/Upgrade_to_Yocto_honister_3.4
sudo apt-get update
sudo apt-get install --yes gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev xterm python3-subunit mesa-common-dev zstd lz4

# Speed up the build and guard against fetcher failures by using Shared State Cache mirrors and enabling Hash Equivalence.
# Use pre-built artifacts rather than building them
# Uncomment the below lines in the local.conf
#   BB_HASHSERVE = "auto"
#   BB_SIGNATURE_HANDLER = "OEEquivHash"
#   BB_HASHSERVE_UPSTREAM = "hashserv.yocto.io:8687"
#   SSTATE_MIRRORS ?= "file://.* https://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH"
nano build/conf/local.conf

# GUI
#   bitbake core-image-sato
# No GUI
bitbake core-image-minimal
echo "=================================================="

echo "# Run the emulator"
# GUI
#   runqemu qemux86-64
# No GUI
runqemu qemux86-64 nographic
# username: root
echo "=================================================="
