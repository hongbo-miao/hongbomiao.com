#!/usr/bin/env bash
set -e

sudo apt-get update
sudo apt-get install --yes golang-go
go install github.com/segmentio/topicctl/cmd/topicctl@latest
