package controllers

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io"
	"net/http"
)

func UpdateSeed(c *gin.Context) {
	bodyBytes, _ := io.ReadAll(c.Request.Body)
	n, err := jsonparser.GetInt(bodyBytes, "input", "n")
	if err != nil {
		log.Error().Err(err).Bytes("bodyBytes", bodyBytes).Msg("jsonparser.GetInt n")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "something bad happened",
		})
		return
	}

	seed, err := utils.SetSeed(int(n))
	if err != nil {
		log.Error().Err(err).Msg("utils.SetSeed")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, seed)
}
