package main

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"io/ioutil"
	"net/http"
)

func logs(c *gin.Context) {
	body, _ := ioutil.ReadAll(c.Request.Body)
	println(string(body))
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}

func main() {
	utils.InitLogger()
	var config = utils.GetConfig()
	log.Info().
		Str("env", config.Env).
		Str("decisionLogReceiverPort", config.DecisionLogReceiverPort).
		Msg("main")

	r := gin.Default()
	r.Use(gzip.Gzip(gzip.DefaultCompression, gzip.WithDecompressFn(gzip.DefaultDecompressHandle)))
	r.POST("/logs", logs)
	_ = r.Run(":" + config.DecisionLogReceiverPort)
}
