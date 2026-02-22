package utils

import (
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	AppEnv string
	Port   string
}

func GetConfig() *Config {
	path := "config/decision_logger/"

	appEnv := os.Getenv("ENVIRONMENT")
	if appEnv == "" {
		appEnv = "development"
	}

	_ = godotenv.Load(path + ".env." + appEnv + ".local")
	_ = godotenv.Load(path + ".env." + appEnv)

	return &Config{
		AppEnv: appEnv,
		Port:   os.Getenv("PORT"),
	}
}
