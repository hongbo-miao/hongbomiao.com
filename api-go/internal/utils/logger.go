package utils

import (
	"go.uber.org/zap"
)

func InitLogger() *zap.Logger {
	var logger, _ = zap.NewProduction()
	defer logger.Sync()
	return logger
}
