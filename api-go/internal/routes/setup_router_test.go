package routes

import (
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/utils"
	"github.com/stretchr/testify/assert"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthRoute(t *testing.T) {
	var config = utils.GetConfig()
	r := SetupRouter(config.Env)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, 200, w.Code)
	assert.Equal(t, "{\"status\":\"ok\"}", w.Body.String())
}
