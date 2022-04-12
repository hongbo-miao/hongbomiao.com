package utils

type Seed struct {
	SeedNumber int `json:"seedNumber"`
}

var seedNumber = 42

func GetSeed() (*Seed, error) {
	return &Seed{
		SeedNumber: seedNumber,
	}, nil
}

func SetSeed(n int) (*Seed, error) {
	seedNumber = n
	return &Seed{
		SeedNumber: seedNumber,
	}, nil
}
