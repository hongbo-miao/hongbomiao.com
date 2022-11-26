#!/usr/bin/env bash
set -e

echo "# Setup minikube"
minikube start --driver=hyperkit --cpus=2 --memory=8g
# minikube delete

minikube addons enable ingress
minikube mount ./data:/data
echo "=================================================="
