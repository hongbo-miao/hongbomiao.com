package utils

import (
	"github.com/golang-jwt/jwt"
	"github.com/rs/zerolog/log"
	"time"
)

type JWT struct {
	JWTToken string `json:"jwtToken"`
}

func GetJWTToken(email string, password string) (JWT, error) {
	var config = GetConfig()

	uid := GetUIDByEmail(email)
	if uid == "" {
		return JWT{}, nil
	}
	isPasswordValid := VerifyPassword(uid, password)
	if !isPasswordValid {
		return JWT{}, nil
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"id":  uid,
		"exp": time.Now().Add(time.Hour * 24).Unix(),
	})

	tokenString, err := token.SignedString([]byte(config.JWTSecret))
	if err != nil {
		log.Error().Err(err).Msg("token.SignedString")
	}

	return JWT{
		JWTToken: tokenString,
	}, nil
}
