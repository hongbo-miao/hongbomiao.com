package handlers

import (
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func PrometheusHandler() gin.HandlerFunc {
	h := promhttp.Handler()
	return gin.WrapH(h)
}
