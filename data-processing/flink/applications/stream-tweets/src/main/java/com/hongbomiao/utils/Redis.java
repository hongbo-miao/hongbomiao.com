package com.hongbomiao.utils;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.connectors.redis.common.mapper.RedisCommand;
import org.apache.flink.streaming.connectors.redis.common.mapper.RedisCommandDescription;
import org.apache.flink.streaming.connectors.redis.common.mapper.RedisMapper;

public class Redis {
  public static class MyRedisMapper implements RedisMapper<Tuple2<String, String>> {
    @Override
    public RedisCommandDescription getCommandDescription() {
      return new RedisCommandDescription(RedisCommand.HSET, "trending-twitter-hashtags");
    }

    @Override
    public String getKeyFromData(Tuple2<String, String> data) {
      return data.f0;
    }

    @Override
    public String getValueFromData(Tuple2<String, String> data) {
      return data.f1;
    }
  }
}
