package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/handlers"
	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
)

func SetupRouter(env string) *gin.Engine {
	r := gin.Default()
	if env == "development" {
		r.Use(handlers.CORSHandler())
	}
	r.Use(otelgin.Middleware("hm-api-server"))
	r.POST("/graphql", handlers.GraphQLHandler())
	r.GET("/", controllers.Health)
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "cool",
		})
	})
	return r
}
