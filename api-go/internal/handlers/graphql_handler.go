package handlers

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/schemas"
	"github.com/gin-gonic/gin"
	"github.com/graphql-go/handler"
)

func GraphQLHandler() gin.HandlerFunc {
	h := handler.New(&handler.Config{
		Schema:   &schemas.Schema,
		Pretty:   true,
		GraphiQL: false,
	})

	return func(c *gin.Context) {
		h.ServeHTTP(c.Writer, c.Request)
	}
}
