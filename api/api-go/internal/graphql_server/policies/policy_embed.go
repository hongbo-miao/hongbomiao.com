package policies

import (
	_ "embed"
)

//go:embed app.rbac.rego
var policy []byte

func ReadPolicy() []byte {
	return policy
}
