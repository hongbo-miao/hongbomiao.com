package controllers

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io/ioutil"
	"net/http"
)

func GetSeed(c *gin.Context) {
	seed, err := utils.GetSeed()
	if err != nil {
		log.Error().Err(err).Msg("utils.GetSeed")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}
	c.JSON(http.StatusOK, seed)
}

func SetSeed(c *gin.Context) {
	bodyBytes, _ := ioutil.ReadAll(c.Request.Body)
	n, err := jsonparser.GetInt(bodyBytes, "n")
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
