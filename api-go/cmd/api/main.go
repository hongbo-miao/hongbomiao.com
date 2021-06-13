package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/handlers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/utils"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func main() {
	var config = utils.InitConfig()
	config.Logger.Info("env",
		zap.String("port", config.Port),
	)

	r := gin.Default()
	r.Use(static.Serve("/", static.LocalFile("./web", true)))
	r.GET("/ping", controllers.Ping)
	r.POST("/graphql", handlers.GraphQLHandler())
	_ = r.Run(":" + config.Port)
}
