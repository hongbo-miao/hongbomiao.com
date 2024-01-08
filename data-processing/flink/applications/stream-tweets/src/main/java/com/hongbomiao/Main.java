package com.hongbomiao;

import com.hongbomiao.utils.Config;
import com.hongbomiao.utils.Redis.MyRedisMapper;
import com.hongbomiao.utils.Tweet;
import com.hongbomiao.utils.TwitterUser;
import java.sql.Timestamp;
import java.util.Objects;
import java.util.Properties;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.connector.jdbc.JdbcConnectionOptions;
import org.apache.flink.connector.jdbc.JdbcExecutionOptions;
import org.apache.flink.connector.jdbc.JdbcSink;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.connectors.redis.RedisSink;
import org.apache.flink.streaming.connectors.redis.common.config.FlinkJedisPoolConfig;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.util.Collector;

public class Main {
  public static void main(String[] args) throws Exception {
    final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

    // String paramsFilePath = "data-processing/flink/applications/stream-tweets/src/main/resources/application-development.properties";
    String paramsFilePath = "application-production.properties";
    Config config = new Config(paramsFilePath);

    Properties props = new Properties();
    props.setProperty(TwitterSource.CONSUMER_KEY, config.twitterAPIKey);
    props.setProperty(TwitterSource.CONSUMER_SECRET, config.twitterAPISecretKey);
    props.setProperty(TwitterSource.TOKEN, config.twitterAccessToken);
    props.setProperty(TwitterSource.TOKEN_SECRET, config.twitterAccessTokenSecret);

    DataStream<String> streamSource = env.addSource(new TwitterSource(props));
    DataStream<JsonNode> tweetStream =
        streamSource
            .flatMap(
                new FlatMapFunction<String, JsonNode>() {
                  @Override
                  public void flatMap(String value, Collector<JsonNode> out) throws Exception {
                    ObjectMapper mapper = new ObjectMapper();
                    JsonNode jsonNode = mapper.readTree(value);
                    out.collect(jsonNode);
                  }
                })
            .filter(
                new FilterFunction<JsonNode>() {
                  @Override
                  public boolean filter(JsonNode value) throws Exception {
                    // value.has("text")
                    // !value.has("retweeted_status")
                    // Objects.equals(value.get("lang").asText(), "en")
                    return value.has("text")
                        && !value.has("retweeted_status")
                        && Objects.equals(value.get("lang").asText(), "en");
                  }
                });

    // Sink to TimescaleDB
    tweetStream
        .map(
            new MapFunction<JsonNode, TwitterUser>() {
              @Override
              public TwitterUser map(JsonNode value) throws Exception {
                return (new TwitterUser(
                    value.get("user").get("id_str").asText(),
                    value.get("user").get("name").asText(),
                    value.get("user").get("profile_image_url_https").asText()));
              }
            })
        .addSink(
            JdbcSink.sink(
                "insert into twitter_user (id, name, avatar) values (?, ?, ?);",
                (statement, tweet) -> {
                  statement.setString(1, tweet.id);
                  statement.setString(2, tweet.name);
                  statement.setString(3, tweet.avatar);
                },
                JdbcExecutionOptions.builder()
                    .withBatchSize(1000)
                    .withBatchIntervalMs(200)
                    .withMaxRetries(5)
                    .build(),
                new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                    .withDriverName("org.postgresql.Driver")
                    .withUrl(config.timescaleDBURL)
                    .withUsername(config.timescaleDBUsername)
                    .withPassword(config.timescaleDBPassword)
                    .build()));

    tweetStream
        .map(
            new MapFunction<JsonNode, Tweet>() {
              @Override
              public Tweet map(JsonNode value) throws Exception {
                return (new Tweet(
                    new Timestamp(Long.parseLong(value.get("timestamp_ms").asText())),
                    value.get("id_str").asText(),
                    value.get("user").get("id_str").asText(),
                    value.get("text").asText(),
                    value.get("lang").asText()));
              }
            })
        .addSink(
            JdbcSink.sink(
                "insert into tweet (timestamp, id, twitter_user_id, text, lang) values (?, ?, ?, ?, ?)",
                (statement, tweet) -> {
                  statement.setTimestamp(1, tweet.timestamp);
                  statement.setString(2, tweet.id);
                  statement.setString(3, tweet.twitterUserID);
                  statement.setString(4, tweet.text);
                  statement.setString(5, tweet.lang);
                },
                JdbcExecutionOptions.builder()
                    .withBatchSize(1000)
                    .withBatchIntervalMs(200)
                    .withMaxRetries(5)
                    .build(),
                new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                    .withDriverName("org.postgresql.Driver")
                    .withUrl(config.timescaleDBURL)
                    .withUsername(config.timescaleDBUsername)
                    .withPassword(config.timescaleDBPassword)
                    .build()));

    // Sink to Redis
    DataStream<Tuple2<String, Integer>> hashtagStream =
        tweetStream.flatMap(
            new FlatMapFunction<JsonNode, Tuple2<String, Integer>>() {
              @Override
              public void flatMap(JsonNode value, Collector<Tuple2<String, Integer>> out)
                  throws Exception {
                for (JsonNode hashtag : value.get("entities").get("hashtags")) {
                  out.collect(new Tuple2<String, Integer>(hashtag.get("text").asText(), 1));
                }
              }
            });

    DataStream<Tuple2<String, Integer>> tweetsPerWindow =
        hashtagStream
            .keyBy(value -> value.f0)
            .window(TumblingProcessingTimeWindows.of(Time.seconds(10)))
            .sum(1);
    tweetsPerWindow.print();

    FlinkJedisPoolConfig conf =
        new FlinkJedisPoolConfig.Builder()
            .setHost(config.redisHost)
            .setPort(config.redisPort)
            .setPassword(config.redisPassword)
            .build();
    tweetsPerWindow
        .map(
            new MapFunction<Tuple2<String, Integer>, Tuple2<String, String>>() {
              @Override
              public Tuple2<String, String> map(Tuple2<String, Integer> value) throws Exception {
                return (new Tuple2<String, String>(value.f0, String.valueOf(value.f1)));
              }
            })
        .addSink(new RedisSink<Tuple2<String, String>>(conf, new MyRedisMapper()));

    env.execute("Get trending Twitter hashtags");
  }
}
