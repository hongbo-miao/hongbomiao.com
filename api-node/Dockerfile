FROM docker.io/node:18.14.0-alpine AS base
WORKDIR /usr/src/app

FROM base AS web
COPY ["web/package.json", "web/package-lock.json", "web/.npmrc", "./"]
RUN npm ci
COPY web ./
# Skip lint check during react-scripts build
RUN rm -f ./.eslintrc.js \
  && npm run build \
  && npm prune --production

FROM base AS api-node
COPY ["api-node/package.json", "api-node/package-lock.json", "api-node/.npmrc", "./"]
RUN npm ci
COPY api-node ./
RUN npm run build \
  && npm prune --production

FROM base AS release
RUN apk add --no-cache dumb-init
ENV NODE_ENV production
USER node
COPY --from=web /usr/src/app/build ./public
COPY --from=api-node /usr/src/app/build ./build
COPY --from=api-node /usr/src/app/node_modules ./node_modules
COPY --from=api-node /usr/src/app/package.json ./
EXPOSE 5000
CMD ["dumb-init", "node", "build/index.js"]
