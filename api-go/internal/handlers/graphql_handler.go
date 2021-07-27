package handlers

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/schemas"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/gin-gonic/gin"
	"github.com/graphql-go/handler"
	"net/http"
)

func addContext(next *handler.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		myID := utils.VerifyToken(r)
		ctx := context.WithValue(r.Context(), types.ContextMyIDKey("myID"), myID)
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
