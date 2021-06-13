package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/config"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/handlers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/utils"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

var app config.AppConfig

func main() {
	const portNumber = ":8080"

	app.Logger = utils.InitLogger()

	r := gin.Default()
	r.Use(static.Serve("/", static.LocalFile("./web", true)))
	r.GET("/ping", controllers.Ping)
	r.POST("/graphql", handlers.GraphQLHandler())
	_ = r.Run(portNumber)

	app.Logger.Info("portNumber",
		// Structured context as strongly typed Field values.
		zap.String("portNumber", portNumber),
	)
}
