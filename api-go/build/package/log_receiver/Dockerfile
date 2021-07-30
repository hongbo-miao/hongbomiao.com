FROM golang:1.16.6-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/log_receiver/ ./cmd/log_receiver/main.go


FROM alpine:3.14.0 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/config/log_receiver/.env.production ./config/log_receiver/.env.production
COPY --from=builder /usr/src/app/out/cmd/log_receiver/main ./main

EXPOSE 8000
CMD ["./main"]
