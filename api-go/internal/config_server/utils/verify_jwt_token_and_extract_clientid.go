package utils

import (
	"errors"
	"github.com/golang-jwt/jwt/v4"
	"github.com/rs/zerolog/log"
	"io/ioutil"
)

type JWTTokenContent struct {
	ID string
}

func VerifyJWTTokenAndExtractClientID(tokenString string) (string, error) {
	config := GetConfig()
	publicKey, err := ioutil.ReadFile(config.OPALAuthPublicKeyPath)
	if err != nil {
		log.Error().Err(err).Msg("ioutil.ReadFile")
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

	clientID := claims["client_id"].(string)
	return clientID, nil
}
