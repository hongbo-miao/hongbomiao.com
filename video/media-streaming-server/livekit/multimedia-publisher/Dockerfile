FROM docker.io/rust:1.96.1-slim-trixie AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    g++ \
    libglib2.0-dev \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --locked --release

FROM docker.io/debian:trixie-slim AS release
WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/multimedia-publisher ./
COPY .env.development ./
COPY .env.production ./
CMD ["./multimedia-publisher"]
