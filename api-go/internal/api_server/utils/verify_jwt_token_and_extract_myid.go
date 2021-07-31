package utils

import (
	"errors"
	"github.com/golang-jwt/jwt"
	"github.com/rs/zerolog/log"
	"net/http"
	"strings"
)

type JWTTokenContent struct {
	ID string
}

func extractBearerToken(r *http.Request) (string, error) {
	bearerToken := r.Header.Get("Authorization")
	strArr := strings.Split(bearerToken, " ")
	if len(strArr) != 2 {
		return "", errors.New("no bearer token")
	}
	return strArr[1], nil
}

func VerifyJWTTokenAndExtractMyID(r *http.Request) (string, error) {
	config := GetConfig()
	tokenString, err := extractBearerToken(r)
	if err != nil {
		log.Error().Err(err).Msg("extractBearerToken")
		return "", err
	}

	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return []byte(config.JWTSecret), nil
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
