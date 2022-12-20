package utils

import (
	"context"
	"encoding/json"
	sq "github.com/Masterminds/squirrel"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog/log"
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

func FetchOPALClientConfig(pg *pgxpool.Pool, opalClientID string) (*OPALClientConfig, error) {
	log.Info().Str("opalClientID", opalClientID).Msg("FetchOPALClientConfig")

	psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)
	sql, args, err := psql.
		Select("config").
		From("opal_client").
		Where("id = ?", opalClientID).
		ToSql()
	if err != nil {
		log.Error().
			Err(err).
			Str("sql", sql).
			Interface("args", args).
			Msg("ToSql")
		return nil, err
	}

	opalClient := new(OPALClient)
	err = pg.QueryRow(context.Background(), sql, args...).Scan(&opalClient.Config)
	if err != nil {
		log.Error().Err(err).Msg("pg.QueryRow")
		return nil, err
	}

	var opalClientConfig OPALClientConfig
	err = json.Unmarshal([]byte(opalClient.Config), &opalClientConfig)
	if err != nil {
		log.Error().Err(err).Msg("json.Unmarshal")
		return nil, err
	}

	return &opalClientConfig, nil
}
