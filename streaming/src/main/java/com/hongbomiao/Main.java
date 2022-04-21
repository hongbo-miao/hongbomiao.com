package com.hongbomiao;

import java.sql.Timestamp;
import java.util.Objects;
import java.util.Properties;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.utils.ParameterTool;
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
import org.apache.flink.streaming.connectors.redis.common.mapper.RedisCommand;
import org.apache.flink.streaming.connectors.redis.common.mapper.RedisCommandDescription;
import org.apache.flink.streaming.connectors.redis.common.mapper.RedisMapper;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.util.Collector;

public class Main {
  static class Tweet {
    public Tweet(Timestamp timestamp, String id, String id_str, String text, String lang) {
      this.timestamp = timestamp;
      this.id = id;
      this.id_str = id_str;
      this.text = text;
      this.lang = lang;
    }

    final Timestamp timestamp;
    final String id;
    final String id_str;
    final String text;
    final String lang;
  }

  public static void main(String[] args) throws Exception {
    final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

    // application-development.properties, application-production.properties
    String propertiesFilePath = "streaming/src/main/resources/application-development.properties";
    ParameterTool parameter = ParameterTool.fromPropertiesFile(propertiesFilePath);
    parameter.getRequired("timescaledb.username");
    System.out.println(parameter.getRequired("timescaledb.username"));

    Properties props = new Properties();
    props.load(TwitterSource.class.getClassLoader().getResourceAsStream("application-development.properties"));
    props.setProperty(TwitterSource.CONSUMER_KEY, props.getProperty("twitter.api.key"));
    props.setProperty(TwitterSource.CONSUMER_SECRET, props.getProperty("twitter.api.secret.key"));
    props.setProperty(TwitterSource.TOKEN, props.getProperty("twitter.access.token"));
    props.setProperty(TwitterSource.TOKEN_SECRET, props.getProperty("twitter.access.token.secret"));

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
            new MapFunction<JsonNode, Tweet>() {
              @Override
              public Tweet map(JsonNode value) throws Exception {
                return (new Tweet(
                    new Timestamp(Long.parseLong(value.get("timestamp_ms").asText())),
                    value.get("id").asText(),
                    value.get("id_str").asText(),
                    value.get("text").asText(),
                    value.get("lang").asText()));
              }
            })
        .addSink(
            JdbcSink.sink(
                "insert into tweets (timestamp, id, id_str, text, lang) values (?, ?, ?, ?, ?)",
                (statement, tweet) -> {
                  statement.setTimestamp(1, tweet.timestamp);
                  statement.setString(2, tweet.id);
                  statement.setString(3, tweet.id_str);
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
                    .withUrl(parameter.getRequired("timescaledb.url"))
                    .withUsername(parameter.getRequired("timescaledb.username"))
                    .withPassword(parameter.getRequired("timescaledb.password"))
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
            .setHost(parameter.getRequired("redis.host"))
            .setPort(Integer.parseInt(parameter.getRequired("redis.port")))
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

  public static class Splitter implements FlatMapFunction<JsonNode, Tuple2<String, Integer>> {
    @Override
    public void flatMap(JsonNode value, Collector<Tuple2<String, Integer>> out) throws Exception {
      for (JsonNode hashtag : value.get("entities").get("hashtags")) {
        out.collect(new Tuple2<String, Integer>(hashtag.get("text").asText(), 1));
      }
    }
  }

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
