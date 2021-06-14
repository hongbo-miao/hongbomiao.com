package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/handlers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

func main() {
	utils.InitLogger()
	var config = utils.InitConfig()

	r := gin.Default()
	r.Use(handlers.CORSHandler())
	r.Use(static.Serve("/", static.LocalFile("./web", true)))
	r.GET("/ping", controllers.Ping)
	r.POST("/graphql", handlers.GraphQLHandler())
	_ = r.Run(":" + config.Port)
}
