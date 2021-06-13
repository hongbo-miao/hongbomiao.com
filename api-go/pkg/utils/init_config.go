package utils

import (
	"github.com/joho/godotenv"
	"go.uber.org/zap"
	"os"
)

type Config struct {
	Logger *zap.Logger
	Port   string
}

func InitConfig() *Config {
	var config Config
	config.Logger = InitLogger()

	err := godotenv.Load()
	if err != nil {
		config.Logger.Fatal("Error loading .env file")
	}
	config.Port = os.Getenv("PORT")
	return &config
}
