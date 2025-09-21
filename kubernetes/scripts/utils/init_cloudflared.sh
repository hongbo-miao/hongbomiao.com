#!/usr/bin/env bash
set -e

echo "# Install cloudflared"
brew install cloudflare/cloudflare/cloudflared
cloudflared tunnel login
echo "=================================================="
