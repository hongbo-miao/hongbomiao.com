#!/usr/bin/env bash
set -e

echo "# Uninstall Postgres"
kubectl delete --filename=kubernetes/manifests/postgres
echo "=================================================="
