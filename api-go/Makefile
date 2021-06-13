run:
	go run cmd/api/main.go

run_grpc:
	go run cmd/grpc/main.go


tidy:
	go mod tidy

go-fmt:
	gofmt .

go-check:
	gofmt -l .

go-lint:
	golangci-lint run ./...

proto-lint:
	buf lint

opa-test:
	opa test internal/policies/*.rego -v

opa-test-coverage:
	opa test internal/policies/*.rego --coverage --format=json

opa-fmt:
	opa fmt internal/policies/*.rego -w

opa-check:
	opa check internal/policies/*.rego

proto:
	protoc api/proto/greet/v1/greet.proto --go_out=plugins=grpc:.
