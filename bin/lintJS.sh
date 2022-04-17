#!/usr/bin/env bash
set -e

# Web
cd web
npm run lint:js

# Mobile
cd ../mobile
npm run lint:js

# API (Node.js)
cd ../api-node
npm run lint:js

# Cypress
cd ../cypress
npm run lint:js

# Ethereum
cd ../ethereum
npm run lint:js
