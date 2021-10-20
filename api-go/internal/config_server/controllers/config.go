package controllers

import (
	"fmt"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_server/utils"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"net/http"
)

func Config(c *gin.Context) {
	token := c.Request.URL.Query().Get("token")
	fmt.Printf("token: %v\n", token)
	myID, err := utils.VerifyJWTTokenAndExtractMyID(token)
	if err != nil {
		log.Error().Err(err).Msg("VerifyJWTTokenAndExtractMyID")
	}
	fmt.Printf("myID: %v\n", myID)
	c.JSON(http.StatusOK, gin.H{
		"status": "config",
	})
}
