package controllers

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/config_server/utils"
	"github.com/gin-gonic/gin"
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

func Config(c *gin.Context) {
	token := c.Request.URL.Query().Get("token")
	clientID, err := utils.VerifyJWTTokenAndExtractClientID(token)
	if err != nil {
		log.Error().Err(err).Msg("VerifyJWTTokenAndExtractMyID")
	}
	entries := map[string]Entry{
		"hm-opal-client": Entry{
			// URL: "postgresql://admin@yb-tservers.yb-operator:5433/opa_db",
			URL: "postgresql://admin@opa-db-service.hm-opa:40072/opa_db",
			Config: EntryConfig{
				Fetcher: "PostgresFetchProvider",
				Query:   "select role, allow from roles;",
				ConnectionParams: EntryConfigConnectionParams{
					Password: "passw0rd",
				},
				DictKey: "role",
			},
			Topics:  []string{"policy_data"},
			DstPath: "roles",
		},
	}
	c.JSON(http.StatusOK, entries[clientID])
}
