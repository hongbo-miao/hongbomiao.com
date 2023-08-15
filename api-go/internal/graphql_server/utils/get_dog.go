package utils

import (
	"errors"
	"github.com/hongbo-miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/rs/zerolog/log"
)

func GetDog(myID string, dogID string) (*types.Dog, error) {
	res, err := GetOPALDecision(myID, "read", "dog")
	if err != nil {
		log.Error().Err(err).Msg("GetDog")
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
