FROM node:16.1.0-alpine AS base
WORKDIR /usr/src/app


FROM base AS web
COPY ["web/package.json", "web/package-lock.json", "./"]
RUN npm ci

COPY web ./
# Skip lint check during react-scripts build
RUN rm -f ./.eslintrc.js
RUN npm run build
RUN npm prune --production


FROM base AS api
COPY ["api/package.json", "api/package-lock.json", "./"]
RUN npm ci
COPY api ./
RUN npm run build
RUN npm prune --production


FROM base AS release
RUN apk add dumb-init
ENV NODE_ENV production
USER node

COPY --from=web /usr/src/app/build ./public
COPY --from=api /usr/src/app/build ./build
COPY --from=api /usr/src/app/node_modules ./node_modules
COPY --from=api /usr/src/app/package.json ./

EXPOSE 5000
CMD ["dumb-init", "node", "build/index.js"]
