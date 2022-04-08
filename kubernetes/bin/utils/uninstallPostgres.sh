#!/usr/bin/env bash
set -e

echo "# Uninstall Postgres"
kubectl delete --filename=kubernetes/config/postgres
echo "=================================================="
