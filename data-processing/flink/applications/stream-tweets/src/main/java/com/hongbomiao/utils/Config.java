package com.hongbomiao.utils;

import java.io.IOException;
import org.apache.flink.api.java.utils.ParameterTool;

public final class Config {
  public Config(String paramsFilePath)
    throws IOException {
    ParameterTool params = ParameterTool.fromPropertiesFile(paramsFilePath);
    this.twitterAPIKey = params.getRequired("twitter.api.key");
    this.twitterAPISecretKey = params.getRequired("twitter.api.secret.key");
    this.twitterAccessToken = params.getRequired("twitter.access.token");
    this.twitterAccessTokenSecret = params.getRequired("twitter.access.token.secret");
    this.timescaleDBURL = params.getRequired("timescaledb.url");
    this.timescaleDBUsername = params.getRequired("timescaledb.username");
    this.timescaleDBPassword = params.getRequired("timescaledb.password");
    this.redisHost = params.getRequired("redis.host");
    this.redisPort = Integer.parseInt(params.getRequired("redis.port"));
    this.redisPassword = params.getRequired("redis.password");
  }

  public final String twitterAPIKey;
  public final String twitterAPISecretKey;
  public final String twitterAccessToken;
  public final String twitterAccessTokenSecret;
  public final String timescaleDBURL;
  public final String timescaleDBUsername;
  public final String timescaleDBPassword;
  public final String redisHost;
  public final int redisPort;
  public final String redisPassword;
}
