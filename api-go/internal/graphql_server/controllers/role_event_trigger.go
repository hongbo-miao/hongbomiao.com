package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io"
	"net/http"
)

func RoleEventTrigger(c *gin.Context) {
	bodyBytes, _ := io.ReadAll(c.Request.Body)
	log.Info().Bytes("bodyBytes", bodyBytes).Msg("RoleEventTrigger")
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}
