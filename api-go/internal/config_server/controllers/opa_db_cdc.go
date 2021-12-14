package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io/ioutil"
	"net/http"
)

func OPADBCDC(c *gin.Context) {
	body, _ := ioutil.ReadAll(c.Request.Body)
	log.Info().Bytes("body", body).Msg("OPADBCDC")
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}
