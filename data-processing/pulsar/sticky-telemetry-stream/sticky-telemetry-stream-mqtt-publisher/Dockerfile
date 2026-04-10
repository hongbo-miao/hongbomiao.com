FROM docker.io/rust:1.94.1-slim-trixie AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install --yes --no-install-recommends pkg-config libssl-dev protobuf-compiler && rm -rf /var/lib/apt/lists/*
COPY proto/ proto/
COPY sticky-telemetry-stream-mqtt-publisher/Cargo.toml sticky-telemetry-stream-mqtt-publisher/Cargo.lock sticky-telemetry-stream-mqtt-publisher/build.rs sticky-telemetry-stream-mqtt-publisher/
COPY sticky-telemetry-stream-mqtt-publisher/src/ sticky-telemetry-stream-mqtt-publisher/src/
RUN cargo build --locked --release --manifest-path sticky-telemetry-stream-mqtt-publisher/Cargo.toml

FROM docker.io/debian:trixie-slim AS release
RUN apt-get update && apt-get install --yes --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/sticky-telemetry-stream-mqtt-publisher/target/release/sticky-telemetry-stream-mqtt-publisher ./
CMD ["./sticky-telemetry-stream-mqtt-publisher"]
