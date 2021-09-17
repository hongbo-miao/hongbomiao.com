package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io/ioutil"
	"net/http"
)

func Logs(c *gin.Context) {
	body, _ := ioutil.ReadAll(c.Request.Body)
	log.Info().Bytes("body", body).Msg("Logs")
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}
