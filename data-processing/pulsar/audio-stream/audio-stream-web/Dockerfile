FROM docker.io/node:24-alpine AS builder
WORKDIR /app
COPY audio-stream-web/package.json audio-stream-web/package-lock.json ./
RUN npm install
COPY audio-stream-web/ .
RUN npm run build

FROM docker.io/caddy:2.11.2-alpine AS release
COPY --from=builder /app/dist /usr/share/caddy
COPY audio-stream-web/Caddyfile /etc/caddy/Caddyfile
EXPOSE 80
