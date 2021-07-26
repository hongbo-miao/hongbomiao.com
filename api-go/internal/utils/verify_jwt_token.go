package utils

import (
	"github.com/golang-jwt/jwt"
	"github.com/rs/zerolog/log"
	"net/http"
	"strings"
)

type JWTTokenContent struct {
	ID string
}

func extractBearerToken(r *http.Request) string {
	bearerToken := r.Header.Get("Authorization")
	strArr := strings.Split(bearerToken, " ")
	if len(strArr) == 2 {
		return strArr[1]
	}
	return ""
}

func VerifyToken(r *http.Request) string {
	var config = GetConfig()
	tokenString := extractBearerToken(r)
	if tokenString == "" {
		log.Info().Msg("No Bearer token.")
		return ""
	}

	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return []byte(config.JWTSecret), nil
	})
	if err != nil {
		log.Error().Err(err).Msg("jwt.Parse")
		return ""
	}

	if !token.Valid {
		log.Error().Msg("token is not valid")
		return ""
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return ""
	}

	id := claims["id"].(string)
	return id
}
