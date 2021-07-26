package handlers

import (
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"time"
)

var devCORSAllowOrigins = map[string]bool{
	"electron://altair":     true,
	"http://localhost:3000": true,
}

func CORSHandler() gin.HandlerFunc {
	return cors.New(cors.Config{
		AllowMethods:     []string{"GET", "POST"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		AllowOriginFunc: func(origin string) bool {
			if !devCORSAllowOrigins[origin] {
				log.Warn().Str("origin", origin).Msg("AllowOriginFunc")
			}
			return devCORSAllowOrigins[origin]
		},
		MaxAge: 12 * time.Hour,
	})
}
