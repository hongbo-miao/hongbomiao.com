run:
	go run cmd/api/main.go

fmt:
	go fmt ./...

tidy:
	go mod tidy

opa-test:
	opa test **/*.rego -v

opa-test-coverage:
	opa test **/*.rego --coverage --format=json

opa-fmt:
	opa fmt pkg/policies/*.rego -w
