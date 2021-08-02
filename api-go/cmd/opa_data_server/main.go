package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/opa_data_server/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/opa_data_server/utils"
	sharedUtils "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/utils"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

func main() {
	sharedUtils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("appEnv", config.AppEnv).
		Str("port", config.Port).
		Str("postgresHost", config.PostgresHost).
		Str("postgresPort", config.PostgresPort).
		Str("postgresDB", config.PostgresDB).
		Str("postgresUser", config.PostgresUser).
		Str("postgresPassword", config.PostgresPassword).
		Msg("main")

	pg := utils.InitPostgres(
		config.PostgresHost,
		config.PostgresPort,
		config.PostgresDB,
		config.PostgresUser,
		config.PostgresPassword)
	defer pg.Close()

	r := gin.Default()
	r.GET("/", controllers.Health)
	r.GET("/data", controllers.Data(pg))
	err := r.Run(":" + config.Port)
	if err != nil {
		log.Error().Err(err).Msg("r.Run")
	}
}
