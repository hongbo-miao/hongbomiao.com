FROM docker.io/node:22.12.0-alpine AS web-base
WORKDIR /usr/src/app

FROM web-base AS web-builder
COPY ["web/package.json", "web/package-lock.json", "web/.npmrc", "./"]
RUN npm ci
COPY web ./
# Skip lint check during react-scripts build
RUN rm -f ./.eslintrc.js \
  && npm run build \
  && npm prune --production

FROM docker.io/node:22.12.0-alpine AS api-node-base
WORKDIR /usr/src/app

FROM api-node-base AS api-node-builder
COPY ["api-node/package.json", "api-node/package-lock.json", "api-node/.npmrc", "./"]
RUN npm ci
COPY api-node ./
COPY --from=web-builder /usr/src/app/build ./public
RUN npm run build \
  && npm prune --production

FROM api-node-base AS release
RUN apk add --no-cache dumb-init
ENV NODE_ENV production
USER node
COPY --from=api-node-builder --chown=node:node /usr/src/app/public ./public
COPY --from=api-node-builder --chown=node:node /usr/src/app/build ./build
COPY --from=api-node-builder --chown=node:node /usr/src/app/node_modules ./node_modules
COPY --from=api-node-builder --chown=node:node /usr/src/app/package.json ./
EXPOSE 58136
CMD ["dumb-init", "node", "build/src/index.js"]
