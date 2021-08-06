FROM golang:1.16.7-alpine AS builder
WORKDIR /usr/src/app

COPY ["api-go/go.mod", "api-go/go.sum", "./"]
RUN go mod download

COPY api-go ./
RUN go build -o ./out/cmd/log_receiver/ ./cmd/log_receiver/main.go


FROM alpine:3.14.1 AS release
WORKDIR /usr/src/app
ENV GIN_MODE=release

COPY --from=builder /usr/src/app/out/cmd/log_receiver/main ./main

EXPOSE 41797
CMD ["./main"]
