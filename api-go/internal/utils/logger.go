package utils

import (
	"go.uber.org/zap"
)

func InitLogger() *zap.Logger {
	var logger, _ = zap.NewProduction()
	defer func(logger *zap.Logger) {
		err := logger.Sync()
		if err != nil {
			logger.Info("logger.Sync",
				zap.NamedError("err", err),
			)
		}
	}(logger)
	return logger
}
