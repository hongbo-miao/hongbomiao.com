package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/handlers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

func main() {
	utils.InitLogger()
	var config = utils.GetConfig()
	log.Info().Str("env", config.Env).Str("port", config.Port).Msg("main")

	r := gin.Default()
	r.Use(handlers.CORSHandler(*config.CORSAllowOrigins))
	r.GET("/ping", controllers.Ping)
	r.POST("/graphql", handlers.GraphQLHandler())
	_ = r.Run(":" + config.Port)
}
