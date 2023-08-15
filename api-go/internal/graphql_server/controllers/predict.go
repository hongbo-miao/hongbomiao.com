package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/rs/zerolog/log"
	"net/http"
)

func Predict(c *gin.Context) {
	fileHeader, err := c.FormFile("file")
	if err != nil {
		log.Error().Err(err).Msg("c.FormFile")
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	prediction, err := utils.GetPrediction(fileHeader)
	if err != nil {
		log.Error().Err(err).Msg("utils.GetPrediction")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, prediction)
}
