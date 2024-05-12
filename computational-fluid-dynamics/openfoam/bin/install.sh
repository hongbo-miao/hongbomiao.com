#!/usr/bin/env bash
set -e

cd "$HOME/bin/"
curl --silent --fail --show-error --location https://develop.openfoam.com/packaging/containers/-/raw/main/openfoam-docker > openfoam-docker
chmod +x openfoam-docker
ln -s -f openfoam-docker openfoam2312-run
