package controllers

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_loader/utils"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog/log"
	"net/http"
)

func Config(pg *pgxpool.Pool) gin.HandlerFunc {
	fn := func(c *gin.Context) {
		token := c.Request.URL.Query().Get("token")
		opalClientID, err := utils.VerifyJWTTokenAndExtractOPALClientID(token)
		if err != nil {
			log.Error().Err(err).Msg("VerifyJWTTokenAndExtractMyID")
		}

		opalClientConfig, err := utils.FetchOPALClientConfig(pg, opalClientID)
		if err != nil {
			log.Error().Err(err).Msg("FetchOPALClientConfig")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		c.JSON(http.StatusOK, opalClientConfig)
	}
	return fn
}
