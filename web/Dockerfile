FROM docker.io/node:22.13.1-alpine AS base
WORKDIR /usr/src/app

FROM base AS builder
COPY ["web/package.json", "web/package-lock.json", "web/.npmrc", "./"]
RUN npm ci
COPY web ./
RUN npm run build \
  && npm prune --production

FROM docker.io/nginx:1.27.3-alpine AS release
COPY --from=builder /usr/src/app/dist /usr/share/nginx/html
COPY web/conf/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD [ "nginx", "-g", "daemon off;" ]
