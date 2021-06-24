package utils

import (
	"github.com/buger/jsonparser"
	"github.com/go-resty/resty/v2"
	"github.com/rs/zerolog/log"
)

type OPAL struct {
	Decision bool `json:"decision"`
}

func GetOPALDecision(user string, action string, object string, resourceType string) (opal OPAL, err error) {
	var config = GetConfig()
	restyClient := resty.New()

	body := map[string]interface{}{
		"input": map[string]interface{}{
			"user":   user,
			"action": action,
			"object": object,
			"type":   resourceType,
		},
	}
	res, err := restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetBody(body).
		Post("http://" + config.OPAHost + ":" + config.OPAPort + "/v1/data/app/rbac/allow")
	if err != nil {
		log.Error().Err(err).Msg("GetOPALDecision")
	}

	decision, err := jsonparser.GetBoolean(res.Body(), "result")
	if err != nil {
		log.Error().Err(err).Msg("GetBoolean")
	}

	return OPAL{
		Decision: decision,
	}, nil
}
