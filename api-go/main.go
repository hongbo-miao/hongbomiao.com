package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/handlers"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()
	router.Use(static.Serve("/", static.LocalFile("./public", true)))
	router.GET("/ping", controllers.Ping)
	router.POST("/graphql", handlers.GraphQLHandler())
	_ = router.Run(":8080")
}
