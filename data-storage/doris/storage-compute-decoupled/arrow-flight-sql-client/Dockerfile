FROM docker.io/rust:1.96.1-slim-trixie AS builder
WORKDIR /build
COPY . .
RUN cargo build --locked --release

FROM docker.io/debian:trixie-slim AS release
COPY --from=builder /build/target/release/arrow-flight-sql-client /usr/local/bin/arrow-flight-sql-client
CMD ["sleep", "infinity"]
