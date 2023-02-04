package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog/log"
	"strconv"
)

func GetTwitterHashtag(text string, rdb *redis.Client) (*types.TwitterHashtag, error) {
	ctx := context.Background()

	res := rdb.HGet(ctx, "trending-twitter-hashtags", text)
	err := res.Err()
	if err == redis.Nil {
		return &types.TwitterHashtag{
			Text:  text,
			Count: 0,
		}, nil
	} else if err != nil {
		log.Error().Err(err).Msg("rdb.HGet")
		return nil, err
	}

	count, err := strconv.Atoi(res.Val())
	if err != nil {
		return nil, err
	}

	return &types.TwitterHashtag{
		Text:  text,
		Count: count,
	}, nil
}
