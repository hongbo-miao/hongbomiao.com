package utils

import (
	"go.uber.org/zap"
)

type Config struct {
	Logger *zap.Logger
}

func InitConfig() Config {
	var config Config
	config.Logger = InitLogger()
	return config
}
