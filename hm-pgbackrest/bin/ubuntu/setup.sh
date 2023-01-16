#!/usr/bin/env bash
set -e

# https://pgbackrest.org/user-guide.html

echo "# Build pgBackRest"
docker create --name=hm-pgbackrest ghcr.io/hongbo-miao/hm-pgbackrest:latest
docker cp hm-pgbackrest:/usr/src/app/build/pgbackrest-release-2.43/src/pgbackrest pgbackrest
echo "=================================================="

echo "# Install pgBackRest"
sudo apt install --yes postgresql-client libxml2
echo "=================================================="
