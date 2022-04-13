#!/usr/bin/env bash
set -e

echo "# Uninstall ORY Hydra"
helm uninstall ory-hydra --namespace=hm-ory-hydra
echo "=================================================="
