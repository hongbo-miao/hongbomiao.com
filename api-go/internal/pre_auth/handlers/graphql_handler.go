package handlers

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/schemas"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/utils"
	"github.com/gin-gonic/gin"
	"github.com/graphql-go/graphql/gqlerrors"
	"github.com/graphql-go/handler"
	"github.com/rs/zerolog/log"
	"net/http"
)

func addContext(next *handler.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		myID, err := utils.VerifyJWTTokenAndExtractMyID(r)
		if err != nil {
			log.Error().Err(err).Msg("VerifyJWTTokenAndExtractMyID")
		}
		ctx := context.WithValue(r.Context(), types.ContextMyIDKey("myID"), myID)
		next.ContextHandler(ctx, w, r)
	})
}

func GraphQLHandler() gin.HandlerFunc {
	h := handler.New(&handler.Config{
		Schema:   &schemas.Schema,
		Pretty:   true,
		GraphiQL: false,
		FormatErrorFn: func(err error) gqlerrors.FormattedError {
			log.Error().Err(err).Msg("FormatErrorFn")
			return gqlerrors.FormatError(err)
		},
	})
	return gin.WrapH(addContext(h))
}
