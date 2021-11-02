package controllers

import (
	"context"
	"encoding/json"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_server/utils"
	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/rs/zerolog/log"
	"net/http"
)

type EntryConfigConnectionParams struct {
	Password string `json:"password"`
}
type EntryConfig struct {
	Fetcher          string                      `json:"fetcher"`
	Query            string                      `json:"query"`
	ConnectionParams EntryConfigConnectionParams `json:"connection_params"`
	DictKey          string                      `json:"dict_key"`
}
type Entry struct {
	URL     string      `json:"url"`
	Config  EntryConfig `json:"config"`
	Topics  []string    `json:"topics"`
	DstPath string      `json:"dst_path"`
}
type OPALClientConfig struct {
	Entries []Entry `json:"entries"`
}
type OPALClient struct {
	OPALClientID string
	Config       string
}

func Config(pg *pgxpool.Pool) gin.HandlerFunc {
	fn := func(c *gin.Context) {
		token := c.Request.URL.Query().Get("token")
		clientID, err := utils.VerifyJWTTokenAndExtractClientID(token)
		if err != nil {
			log.Error().Err(err).Msg("VerifyJWTTokenAndExtractMyID")
		}

		psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)
		sql, args, err := psql.
			Select("config").
			From("opal_clients").
			Where("opal_client_id = ?", clientID).
			ToSql()
		if err != nil {
			log.Error().
				Err(err).
				Str("sql", sql).
				Interface("args", args).
				Msg("ToSql")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		opalClient := new(OPALClient)
		err = pg.QueryRow(context.Background(), sql, args...).Scan(&opalClient.Config)
		if err != nil {
			log.Error().Err(err).Msg("pg.QueryRow")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		var opalClientConfig OPALClientConfig
		err = json.Unmarshal([]byte(opalClient.Config), &opalClientConfig)
		if err != nil {
			log.Error().Err(err).Msg("pg.QueryRow")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		c.JSON(http.StatusOK, opalClientConfig)
	}
	return fn
}
