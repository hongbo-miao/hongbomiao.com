package utils

import (
	"errors"
	"github.com/golang-jwt/jwt/v4"
	"github.com/rs/zerolog/log"
	"os"
)

type JWTTokenContent struct {
	ID string
}

func VerifyJWTTokenAndExtractOPALClientID(tokenString string) (string, error) {
	config := GetConfig()
	publicKey, err := os.ReadFile(config.OPALAuthPublicKeyPath)
	if err != nil {
		log.Error().Err(err).Msg("os.ReadFile")
		return "", err
	}
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return jwt.ParseRSAPublicKeyFromPEM([]byte(publicKey))
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

	opalClientID := claims["opal_client_id"].(string)
	return opalClientID, nil
}
