package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/handlers"
	"github.com/gin-contrib/logger"
	"github.com/gin-gonic/gin"
	"go.elastic.co/apm/module/apmgin"
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
)

func SetupRouter(env string) *gin.Engine {
	r := gin.New()
	r.Use(apmgin.Middleware(r))
	if env == "development" {
		r.Use(handlers.CORSHandler())
	}
	r.Use(logger.SetLogger())
	r.Use(otelgin.Middleware("hm_pre_auth"))
	r.POST("/graphql", handlers.GraphQLHandler())
	r.GET("/", controllers.Health)
	return r
}
