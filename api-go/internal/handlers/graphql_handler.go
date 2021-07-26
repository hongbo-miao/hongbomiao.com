package handlers

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/schemas"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/gin-gonic/gin"
	"github.com/graphql-go/handler"
	"net/http"
)

type ContextMyIDKey string

func addContext(next *handler.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		myID := utils.VerifyToken(r)
		ctx := context.WithValue(r.Context(), ContextMyIDKey("myID"), myID)
		next.ContextHandler(ctx, w, r)
	})
}

func GraphQLHandler() gin.HandlerFunc {
	h := handler.New(&handler.Config{
		Schema:   &schemas.Schema,
		Pretty:   true,
		GraphiQL: false,
	})
	return gin.WrapH(addContext(h))
}
