#!/usr/bin/env bash
set -e

minikube start --driver=hyperkit --cpus=2 --memory=8g
# minikube delete

minikube addons enable ingress
minikube mount ./kubernetes/data:/data
