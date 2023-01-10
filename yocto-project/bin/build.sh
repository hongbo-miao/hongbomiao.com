#!/usr/bin/env bash
set -e

echo "# Clone Poky"
sudo apt install --yes git
git clone git://git.yoctoproject.org/poky
cd poky
git checkout kirkstone
source oe-init-build-env

echo "# Build"
# Install chrpath, diffstat, lz4c
# https://wiki.koansoftware.com/index.php/Upgrade_to_Yocto_honister_3.4
sudo apt install --yes chrpath diffstat lz4

# Speed up the build and guard against fetcher failures by using Shared State Cache mirrors and enabling Hash Equivalence.
# Use pre-built artifacts rather than building them
# Uncomment the below lines in the local.conf
#   BB_SIGNATURE_HANDLER = "OEEquivHash"
#   BB_HASHSERVE = "auto"
#   BB_HASHSERVE_UPSTREAM = "hashserv.yocto.io:8687"
#   SSTATE_MIRRORS ?= "file://.* https://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH"
nano build/conf/local.conf

bitbake core-image-sato
echo "=================================================="

echo "# Simulate the image"
runqemu qemux86-64
echo "=================================================="
