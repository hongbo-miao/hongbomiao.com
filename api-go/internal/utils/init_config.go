package utils

import (
	"github.com/joho/godotenv"
	"os"
)

type Config struct {
	Port string
	Env  string
}

func InitConfig() *Config {
	env := os.Getenv("APP_ENV")
	if env == "" {
		env = "development"
	}

	_ = godotenv.Load(".env." + env + ".local")
	_ = godotenv.Load(".env." + env)
	_ = godotenv.Load() // .env

	return &Config{
		Env:  env,
		Port: os.Getenv("PORT"),
	}
}
