package utils

import (
	"github.com/go-resty/resty/v2"
	"github.com/rs/zerolog/log"
)

func ConfigureOPALClient(opalClientConfig *OPALClientConfig) ([]byte, error) {
	config := GetConfig()
	restyClient := resty.New()
	res, err := restyClient.R().
		SetHeader("Content-Type", "application/json").
		SetAuthToken(config.OPALClientToken).
		SetBody(opalClientConfig).
		Post("http://" + config.OPALServerHost + ":" + config.OPALServerPort + "/data/config")
	if err != nil {
		log.Error().Err(err).Msg("ConfigureOPALClient")
		return nil, err
	}
	return res.Body(), nil
}
