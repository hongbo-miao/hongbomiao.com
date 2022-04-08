package utils

import (
	"time"
)

type CurrentTime struct {
	Now string `json:"now"`
}

func GetCurrentTime() (*CurrentTime, error) {
	now := time.Now()
	return &CurrentTime{
		Now: now.String(),
	}, nil
}
