package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	AppEnv           string
	Port             string
	PostgresHost     string
	PostgresPort     string
	PostgresDatabase string
	PostgresUser     string
	PostgresPassword string
}

func GetConfig() *Config {
	path := "config/opa_data_server/"

	appEnv := os.Getenv("APP_ENV")
	if appEnv == "" {
		appEnv = "development"
	}

	_ = godotenv.Load(path + ".env." + appEnv + ".local")
	_ = godotenv.Load(path + ".env." + appEnv)

	return &Config{
		AppEnv:           appEnv,
		Port:             os.Getenv("PORT"),
		PostgresHost:     os.Getenv("POSTGRES_HOST"),
		PostgresPort:     os.Getenv("POSTGRES_PORT"),
		PostgresDatabase: os.Getenv("POSTGRES_DATABASE"),
		PostgresUser:     os.Getenv("POSTGRES_USER"),
		PostgresPassword: os.Getenv("POSTGRES_PASSWORD"),
	}
}
