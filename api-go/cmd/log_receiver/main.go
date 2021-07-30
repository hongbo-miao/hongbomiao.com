package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/log_receiver/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/log_receiver/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

func main() {
	sharedUtils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("appEnv", config.AppEnv).
		Str("port", config.Port).
		Msg("main")

	r := gin.Default()
	r.Use(gzip.Gzip(gzip.DefaultCompression, gzip.WithDecompressFn(gzip.DefaultDecompressHandle)))
	r.POST("/logs", controllers.Logs)
	_ = r.Run(":" + config.Port)
}
