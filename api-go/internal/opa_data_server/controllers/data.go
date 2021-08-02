package controllers

import (
	"context"
	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/rs/zerolog/log"
	"net/http"
)

type ActionType struct {
	Action string `json:"action,omitempty"`
	Type   string `json:"type,omitempty"`
}
type DataJSONB struct {
	UserRoles  map[string][]string `json:"user_roles,omitempty"`
	UserGrants struct {
		Customer []ActionType `json:"customer,omitempty"`
		Employee []ActionType `json:"employee,omitempty"`
		Billing  []ActionType `json:"billing,omitempty"`
	} `json:"role_grants,omitempty"`
}
type OPA struct {
	ID   string
	Org  string
	Data DataJSONB
}

func Data(pg *pgxpool.Pool) gin.HandlerFunc {
	fn := func(c *gin.Context) {
		params := c.Request.URL.Query()
		org := params["org"][0]

		psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)
		sql, args, err := psql.
			Select("data").
			From("opa").
			Where("org = ?", org).
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

		opa := new(OPA)
		err = pg.QueryRow(context.Background(), sql, args...).Scan(&opa.Data)
		if err != nil {
			log.Error().Err(err).Msg("pg.QueryRow")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "something bad happened",
			})
			return
		}

		c.JSON(http.StatusOK, opa.Data)
	}
	return fn
}
