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
	const portNumber = ":8080"

	var config = utils.InitConfig()

	r := gin.Default()
	r.Use(static.Serve("/", static.LocalFile("./web", true)))
	r.GET("/ping", controllers.Ping)
	r.POST("/graphql", handlers.GraphQLHandler())
	_ = r.Run(portNumber)

	config.Logger.Info("portNumber",
		// Structured context as strongly typed Field values.
		zap.String("portNumber", portNumber),
	)
}
