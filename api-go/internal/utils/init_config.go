package utils

import (
	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
	"os"
)

type Config struct {
	Port string
}

func InitConfig() *Config {
	var config Config

	err := godotenv.Load()
	if err != nil {
		log.Fatal().Msg("Error loading .env file.")
	}

	config.Port = os.Getenv("PORT")
	return &config
}
