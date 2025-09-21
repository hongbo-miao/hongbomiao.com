#!/usr/bin/env bash
set -e

# https://docs.px.dev/installing-pixie/install-guides/community-cloud-for-pixie/

echo "# Install the Pixie CLI"
bash -c "$(curl --silent --fail --show-error --location https://withpixie.ai/install.sh)"
echo "=================================================="

echo "# Deploy Pixie"
px deploy
echo "=================================================="
