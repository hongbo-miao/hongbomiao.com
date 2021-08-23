package utils

import (
	"errors"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/pre_auth/utils"
	"github.com/rs/zerolog/log"
)

func AdoptDog(myID string, dogID string) (*types.Dog, error) {
	res, err := utils.GetOPALDecision(myID, "adopt", "dog")
	if err != nil {
		log.Error().Err(err).Msg("GetOPALDecision")
		return nil, err
	}
	if !res.Decision {
		return nil, errors.New("no permission")
	}
	return &types.Dog{
		ID:   dogID,
		Name: "Bella",
	}, nil
}
