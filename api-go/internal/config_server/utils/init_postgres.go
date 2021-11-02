package utils

import (
	"context"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/rs/zerolog/log"
	"os"
)

func InitPostgres(
	postgresHost string,
	postgresPort string,
	postgresDB string,
	postgresUser string,
	postgresPassword string) *pgxpool.Pool {
	databaseURL := "postgres://" + postgresUser + ":" + postgresPassword + "@" + postgresHost + ":" + postgresPort + "/" + postgresDB
	pg, err := pgxpool.Connect(context.Background(), databaseURL)
	if err != nil {
		log.Error().Err(err).Msg("conn.Close")
		os.Exit(1)
	}
	return pg
}
