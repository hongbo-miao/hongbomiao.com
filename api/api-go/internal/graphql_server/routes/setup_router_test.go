package routes

import (
	"github.com/hongbo-miao/hongbomiao.com/api/api-go/internal/graphql_server/utils"
	"github.com/stretchr/testify/assert"
	"github.com/valkey-io/valkey-go/mock"
	"go.uber.org/mock/gomock"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthRoute(t *testing.T) {
	config := utils.GetConfig()
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	valkeyClient := mock.NewClient(ctrl)
	r := SetupRouter(config.AppEnv, valkeyClient, nil)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, 200, w.Code)
	assert.Equal(t, "{\"status\":\"ok\"}", w.Body.String())
}
