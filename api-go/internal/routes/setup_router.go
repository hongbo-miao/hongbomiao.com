package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/handlers"
	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
)

func SetupRouter() *gin.Engine {
	r := gin.Default()
	r.Use(otelgin.Middleware("hm-api-server"))
	r.POST("/graphql", handlers.GraphQLHandler())
	r.GET("/", controllers.Health)
	return r
}
