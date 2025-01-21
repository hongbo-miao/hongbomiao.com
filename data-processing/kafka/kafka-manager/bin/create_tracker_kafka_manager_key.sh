#!/usr/bin/env bash
set -e

ssh-keygen -t ed25519 -f ~/.ssh/hm-development-tracker-kafka-manager-key -C hm-development-tracker-kafka-manager
ssh-keygen -t ed25519 -f ~/.ssh/hm-production-tracker-kafka-manager-key -C hm-production-tracker-kafka-manager
