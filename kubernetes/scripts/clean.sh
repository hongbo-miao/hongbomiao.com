#!/usr/bin/env bash
set -e

k3d cluster delete west
k3d cluster delete east
