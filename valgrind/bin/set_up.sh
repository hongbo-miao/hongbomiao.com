#!/usr/bin/env bash
set -e

# https://valgrind.org/downloads
VALGRIND_VERSION=3.22.0
curl --silent --fail --show-error --location "https://sourceware.org/pub/valgrind/valgrind-${VALGRIND_VERSION}.tar.bz2" | tar --extract --bzip2 --verbose
cd "valgrind-${VALGRIND_VERSION}"
export CFLAGS="-march=native"
./configure
make
sudo make install
