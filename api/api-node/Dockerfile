FROM docker.io/node:22.21.1-trixie-slim AS base
WORKDIR /usr/src/app

FROM base AS web-builder
COPY ["web/package.json", "web/package-lock.json", "web/.npmrc", "./"]
RUN npm ci
COPY web ./
RUN npm run build \
  && npm prune --production

FROM base AS api-node-builder
COPY ["api/api-node/package.json", "api/api-node/package-lock.json", "api/api-node/.npmrc", "./"]
RUN npm ci
COPY api/api-node ./
COPY --from=web-builder /usr/src/app/dist ./public
RUN npm run build \
  && npm prune --production

FROM base AS release
RUN apt-get update \
  && apt-get install --yes --no-install-recommends dumb-init \
  && rm -r -f /var/lib/apt/lists/*
ENV NODE_ENV=production
USER node
COPY --from=api-node-builder --chown=node:node /usr/src/app/public ./public
COPY --from=api-node-builder --chown=node:node /usr/src/app/build ./build
COPY --from=api-node-builder --chown=node:node /usr/src/app/node_modules ./node_modules
COPY --from=api-node-builder --chown=node:node /usr/src/app/package.json ./
EXPOSE 58136
CMD ["dumb-init", "node", "build/src/index.js"]
