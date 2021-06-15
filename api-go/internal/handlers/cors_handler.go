package handlers

import (
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"time"
)

func CORSHandler(allowOrigins map[string]bool) gin.HandlerFunc {
	return cors.New(cors.Config{
		AllowMethods:     []string{"GET", "POST"},
		AllowHeaders:     []string{"Origin"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		AllowOriginFunc: func(origin string) bool {
			if !allowOrigins[origin] {
				log.Warn().Str("origin", origin).Send()
			}
			return allowOrigins[origin]
		},
		MaxAge: 12 * time.Hour,
	})
}
