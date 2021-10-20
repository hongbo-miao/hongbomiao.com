package utils

import (
	"errors"
	"github.com/golang-jwt/jwt/v4"
	"github.com/rs/zerolog/log"
)

type JWTTokenContent struct {
	ID string
}

func VerifyJWTTokenAndExtractMyID(tokenString string) (string, error) {
	config := GetConfig()

	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return []byte(config.OPALAuthPublicKey), nil
	})
	if err != nil {
		log.Error().Err(err).Msg("jwt.Parse")
		return "", err
	}

	if !token.Valid {
		log.Error().Msg("token is not valid")
		return "", err
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return "", errors.New("token.Claims")
	}

	id := claims["id"].(string)
	return id, nil
}
