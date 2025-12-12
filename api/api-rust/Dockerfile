FROM docker.io/rust:1.92.0-alpine AS base
WORKDIR /usr/src/app

FROM base AS builder
RUN apk add --no-cache musl-dev
COPY api/api-rust/Cargo.toml api/api-rust/Cargo.lock ./
COPY api/api-rust/src/ src/
RUN cargo build --release

FROM base AS release
COPY --from=builder /usr/src/app/target/release/api-rust ./
COPY api/api-rust/.env.production /usr/src/app/.env.production
EXPOSE 36147
CMD ["./api-rust"]
