FROM docker.io/golang:1.23.4-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/decision_logger/ ./cmd/decision_logger/main.go


FROM docker.io/alpine:3.21.2 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/decision_logger/main ./main

EXPOSE 41797
CMD ["./main"]
