package handlers

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/schemas"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/utils"
	"github.com/gin-gonic/gin"
	"github.com/graphql-go/graphql/gqlerrors"
	"github.com/graphql-go/handler"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog/log"
	"net/http"
)

func addContext(next *handler.Handler, rdb *redis.Client) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		myID, err := utils.VerifyJWTTokenAndExtractMyID(r)
		if err != nil {
			log.Error().Err(err).Msg("VerifyJWTTokenAndExtractMyID")
		}
		ctx := context.WithValue(r.Context(), types.ContextKey("myID"), myID)
		ctx = context.WithValue(ctx, types.ContextKey("rdb"), rdb)
		next.ContextHandler(ctx, w, r)
	})
}

func GraphQLHandler(rdb *redis.Client) gin.HandlerFunc {
	h := handler.New(&handler.Config{
		Schema:   &schemas.Schema,
		Pretty:   true,
		GraphiQL: false,
		FormatErrorFn: func(err error) gqlerrors.FormattedError {
			log.Error().Err(err).Msg("FormatErrorFn")
			return gqlerrors.FormatError(err)
		},
	})
	return gin.WrapH(addContext(h, rdb))
}
