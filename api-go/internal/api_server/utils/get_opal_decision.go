package utils

import (
	"github.com/buger/jsonparser"
	"github.com/go-resty/resty/v2"
	"github.com/rs/zerolog/log"
)

type OPAL struct {
	Decision bool `json:"decision"`
}

func GetOPALDecision(uid string, action string, resourceType string) (*OPAL, error) {
	var config = GetConfig()
	restyClient := resty.New()

	body := map[string]interface{}{
		"input": map[string]interface{}{
			"uid":    uid,
			"action": action,
			"type":   resourceType,
		},
	}
	res, err := restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetBody(body).
		Post("http://" + config.OPAHost + ":" + config.OPAPort + "/v1/data/app/rbac/allow")
	if err != nil {
		log.Error().Err(err).Msg("GetOPALDecision")
		return nil, err
	}

	decision, err := jsonparser.GetBoolean(res.Body(), "result")
	if err != nil {
		log.Error().Err(err).Msg("jsonparser.GetBoolean")
		return nil, err
	}

	return &OPAL{
		Decision: decision,
	}, nil
}
