#!/usr/bin/env bash
set -e

echo "# Install Alpine"
kubectl apply --filename=kubernetes/config/alpine
echo "=================================================="
