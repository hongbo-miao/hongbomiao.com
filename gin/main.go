package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/gin/handlers"
	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()
	router.Use(static.Serve("/", static.LocalFile("./public", true)))
	router.GET("/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "pong",
		})
	})
	router.POST("/graphql", handlers.GraphQLHandler())
	_ = router.Run(":8080")
}
