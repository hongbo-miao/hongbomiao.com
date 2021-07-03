package routes

import (
	"contrib.go.opencensus.io/exporter/ocagent"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/controllers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/handlers"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"go.opencensus.io/trace"
	"time"
)

func SetupRouter() *gin.Engine {
	var config = utils.GetConfig()
	oce, err := ocagent.NewExporter(
		ocagent.WithInsecure(),
		ocagent.WithReconnectionPeriod(5*time.Second),
		ocagent.WithAddress(config.OpenCensusAgentHost+":"+config.OpenCensusAgentPort),
		ocagent.WithServiceName("api-go"))
	if err != nil {
		log.Error().Err(err).Msg("ocagent.NewExporter")
	}
	trace.RegisterExporter(oce)

	r := gin.Default()
	r.POST("/graphql", handlers.GraphQLHandler())
	r.GET("/", controllers.Health)
	return r
}
