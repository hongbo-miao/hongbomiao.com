package utils

import (
	"errors"
	"github.com/rs/zerolog/log"
)

type JWT struct {
	JWTToken string `json:"jwtToken"`
}

func SignIn(email string, password string) (*JWT, error) {
	user, err := GetUserByEmail(email)
	if err != nil {
		log.Error().Err(err).Msg("GetUserByEmail")
		return nil, err
	}

	isPasswordValid, err := VerifyPassword(user.ID, password)
	if err != nil {
		log.Error().Err(err).Msg("VerifyPassword")
		return nil, err
	}
	if !isPasswordValid {
		return nil, errors.New("not valid password")
	}

	jwtToken, err := GenerateJWTToken(user.ID)
	if err != nil {
		log.Error().Err(err).Msg("GenerateJWTToken")
		return nil, err
	}

	return &JWT{
		JWTToken: jwtToken,
	}, nil
}
