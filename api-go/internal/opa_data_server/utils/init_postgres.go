package utils

import (
	"context"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/rs/zerolog/log"
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
		return nil
	}
	// defer pg.Close()
	return pg
}
