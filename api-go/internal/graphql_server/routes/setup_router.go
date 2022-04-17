package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/handlers"
	sharedControllers "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/controllers"
	sharedHandlers "github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/shared/handlers"
	"github.com/gin-contrib/logger"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/minio/minio-go/v7"
	"go.elastic.co/apm/module/apmgin"
	"go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
)

func SetupRouter(env string, rdb *redis.Client, minioClient *minio.Client) *gin.Engine {
	r := gin.New()
	r.Use(apmgin.Middleware(r))
	if env == "development" {
		r.Use(handlers.CORSHandler())
	}
	r.Use(logger.SetLogger())
	r.Use(otelgin.Middleware("hm-graphql-server"))
	r.GET("/", sharedControllers.Health)
	r.GET("/metrics", sharedHandlers.PrometheusHandler())
	r.POST("/graphql", handlers.GraphQLHandler(rdb))
	r.POST("/hasura/update-seed", controllers.UpdateSeed)
	r.POST("/predict", controllers.Predict)
	r.POST("/upload", controllers.Upload(minioClient))
	return r
}
