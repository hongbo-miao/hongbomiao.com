package handlers

import (
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"time"
)

func CORSHandler() gin.HandlerFunc {
	return cors.New(cors.Config{
		AllowOrigins:     []string{"https://www.hongbomiao.com"},
		AllowMethods:     []string{"GET", "POST"},
		AllowHeaders:     []string{"Origin"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		AllowOriginFunc: func(origin string) bool {
			log.Warn().Str("origin", origin).Send()
			return origin == "electron://altair"
		},
		MaxAge: 12 * time.Hour,
	})
}
