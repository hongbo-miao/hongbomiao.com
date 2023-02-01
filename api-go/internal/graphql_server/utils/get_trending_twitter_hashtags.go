package utils

import (
	"context"
	"github.com/Hongbo-Miao/hongbomiao.com/api-go/internal/graphql_server/types"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog/log"
	"strconv"
)

func GetTrendingTwitterHashtags(rdb *redis.Client) ([]types.TwitterHashtag, error) {
	ctx := context.Background()
	var trendingTwitterHashtags []types.TwitterHashtag

	res := rdb.HGetAll(ctx, "trending-twitter-hashtags")
	err := res.Err()
	if err == redis.Nil {
		log.Error().Err(err).Msg("rdb.HGet")
		return trendingTwitterHashtags, nil
	} else if err != nil {
		log.Error().Err(err).Msg("rdb.HGet")
		return nil, err
	}

	for k, v := range res.Val() {
		count, err := strconv.Atoi(v)
		if err != nil {
			return nil, err
		}
		trendingTwitterHashtags = append(trendingTwitterHashtags, types.TwitterHashtag{
			Text:  k,
			Count: count,
		})
	}

	return trendingTwitterHashtags, nil
}
