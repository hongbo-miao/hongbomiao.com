package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	Port                string
	Env                 string
	GRPCHost            string
	GRPCPort            string
	OPAHost             string
	OPAPort             string
	OpenCensusAgentHost string
	OpenCensusAgentPort string
}

func GetConfig() *Config {
	env := os.Getenv("APP_ENV")
	if env == "" {
		env = "development"
	}

	_ = godotenv.Load(".env." + env + ".local")
	_ = godotenv.Load(".env." + env)
	_ = godotenv.Load() // .env

	return &Config{
		Env:                 env,
		Port:                os.Getenv("PORT"),
		GRPCHost:            os.Getenv("GRPC_HOST"),
		GRPCPort:            os.Getenv("GRPC_PORT"),
		OPAHost:             os.Getenv("OPA_HOST"),
		OPAPort:             os.Getenv("OPA_PORT"),
		OpenCensusAgentHost: os.Getenv("OPEN_CENSUS_AGENT_HOST"),
		OpenCensusAgentPort: os.Getenv("OPEN_CENSUS_AGENT_PORT"),
	}
}
