package policies

import (
	_ "embed"
)

//go:embed data.json
var data []byte

func ReadData() []byte {
	return data
}
