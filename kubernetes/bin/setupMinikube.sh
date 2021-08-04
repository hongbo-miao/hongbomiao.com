#!/usr/bin/env bash

set -e

minikube config set cpus 4
minikube config set memory 8192
minikube start --driver=hyperkit
minikube addons enable ingress
minikube mount ./kubernetes/data:/data
