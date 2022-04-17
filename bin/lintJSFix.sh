#!/usr/bin/env bash
set -e

# Web
cd web
npm run lint:js:fix

# Mobile
cd ../mobile
npm run lint:js:fix

# API (Node.js)
cd ../api-node
npm run lint:js:fix

# Cypress
cd ../cypress
npm run lint:js:fix

# Ethereum
cd ../ethereum
npm run lint:js:fix
