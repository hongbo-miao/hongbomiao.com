package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/api_server/utils"
	"github.com/go-redis/redismock/v8"
	"github.com/stretchr/testify/assert"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthRoute(t *testing.T) {
	var config = utils.GetConfig()
	rdb, _ := redismock.NewClientMock()
	r := SetupRouter(config.AppEnv, rdb)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, 200, w.Code)
	assert.Equal(t, "{\"status\":\"ok\"}", w.Body.String())
}
