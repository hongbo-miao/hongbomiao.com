package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/pkg/handlers"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()
	router.Use(static.Serve("/", static.LocalFile("./web", true)))
	router.GET("/ping", controllers.Ping)
	router.POST("/graphql", handlers.GraphQLHandler())
	_ = router.Run(":8080")
}
