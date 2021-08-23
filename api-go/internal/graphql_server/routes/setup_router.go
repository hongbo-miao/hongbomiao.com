package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/handlers"
	"github.com/gin-contrib/logger"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"go.elastic.co/apm/module/apmgin"
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
)

func SetupRouter(env string, rdb *redis.Client) *gin.Engine {
	r := gin.New()
	r.Use(apmgin.Middleware(r))
	if env == "development" {
		r.Use(handlers.CORSHandler())
	}
	r.Use(logger.SetLogger())
	r.Use(otelgin.Middleware("hm-graphql-server"))
	r.POST("/graphql", handlers.GraphQLHandler(rdb))
	r.GET("/", controllers.Health)
	return r
}
