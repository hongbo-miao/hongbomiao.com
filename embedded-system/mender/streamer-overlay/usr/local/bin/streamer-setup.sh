#!/usr/bin/env bash
set -euo pipefail

marker=/opt/streamer/.deps-installed
if [ -f "$marker" ]; then
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

for attempt in $(seq 1 30); do
  if apt-get update && apt-get install --yes --no-install-recommends \
      libsoapysdr0.8 \
      soapysdr-module-lms7 \
      soapysdr-module-airspy \
      soapysdr-tools; then
    touch "$marker"
    exit 0
  fi
  echo "streamer-setup: dependency install attempt ${attempt} failed; retrying in 10s" >&2
  sleep 10
done

echo "streamer-setup: failed to install runtime dependencies after 30 attempts" >&2
exit 1
