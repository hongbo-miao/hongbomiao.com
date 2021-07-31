package utils

import (
	"github.com/golang-jwt/jwt"
	"github.com/rs/zerolog/log"
	"time"
)

func GenerateJWTToken(uid string) (string, error) {
	config := GetConfig()

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"id":  uid,
		"exp": time.Now().Add(time.Hour * 24).Unix(),
	})

	tokenString, err := token.SignedString([]byte(config.JWTSecret))
	if err != nil {
		log.Error().Err(err).Msg("token.SignedString")
		return "", err
	}

	return tokenString, nil
}
