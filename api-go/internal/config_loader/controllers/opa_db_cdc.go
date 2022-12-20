package controllers

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_loader/utils"
	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog/log"
	"io"
	"net/http"
)

func OPADBCDC(pg *pgxpool.Pool) gin.HandlerFunc {
	fn := func(c *gin.Context) {
		bodyBytes, _ := io.ReadAll(c.Request.Body)
		opalClientID, err := jsonparser.GetString(bodyBytes, "after", "opal_client_id")
		if err != nil {
			log.Error().Err(err).Bytes("bodyBytes", bodyBytes).Msg("jsonparser.GetString opal_client_id")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		opalClientConfig, err := utils.FetchOPALClientConfig(pg, opalClientID)
		if err != nil {
			log.Error().Err(err).Msg("FetchOPALClientConfig")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}
		log.Info().Interface("opalClientConfig", opalClientConfig).Msg("FetchOPALClientConfig")

		_, err = utils.ConfigureOPALClient(opalClientConfig)
		if err != nil {
			log.Error().Err(err).Msg("ConfigureOPALClient")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status": "ok",
		})
	}
	return fn
}
