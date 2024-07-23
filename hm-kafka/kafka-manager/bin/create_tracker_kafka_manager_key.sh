#!/usr/bin/env bash
set -e

ssh-keygen -t ed25519 -f ~/.ssh/horizon-development-tracker-kafka-manager-key -C horizon-development-tracker-kafka-manager
ssh-keygen -t ed25519 -f ~/.ssh/horizon-production-tracker-kafka-manager-key -C horizon-production-tracker-kafka-manager
