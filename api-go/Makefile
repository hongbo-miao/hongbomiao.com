run:
	go run cmd/api/main.go

run_grpc:
	go run cmd/grpc/main.go

fmt:
	go fmt ./...

tidy:
	go mod tidy

lint-go:
	golangci-lint run ./...

lint-proto:
	buf lint

opa-test:
	opa test pkg/policies/*.rego -v

opa-test-coverage:
	opa test pkg/policies/*.rego --coverage --format=json

opa-fmt:
	opa fmt pkg/policies/*.rego -w

proto:
	protoc api/proto/greet/v1/greet.proto --go_out=plugins=grpc:.
