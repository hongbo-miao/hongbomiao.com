package utils

import (
	"github.com/bep/debounce"
	"go.uber.org/atomic"
	"time"
)

type DebouncedSeed struct {
	DebouncedSeedNumber int `json:"debouncedSeedNumber"`
}

func GetDebouncedSeed() (*Seed, error) {
	counter := atomic.NewUint64(0)
	f := func() {
		counter.Add(42)
	}
	debounced := debounce.New(100 * time.Millisecond)
	for i := 0; i < 10; i++ {
		debounced(f)
	}
	time.Sleep(200 * time.Millisecond)
	return &Seed{
		SeedNumber: int(counter.Load()),
	}, nil
}
