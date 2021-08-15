package com.hongbomiao;

import java.util.Objects;
import java.util.Properties;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.util.Collector;

public class Main {
  public static void main(String[] args) throws Exception {
    final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

    Properties props = new Properties();
    props.load(TwitterSource.class.getClassLoader().getResourceAsStream("application.properties"));
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
    env.execute("Find trending Twitter hashtags");
  }

  public static class Splitter implements FlatMapFunction<JsonNode, Tuple2<String, Integer>> {
    @Override
    public void flatMap(JsonNode value, Collector<Tuple2<String, Integer>> out) throws Exception {
      for (JsonNode hashtag : value.get("entities").get("hashtags")) {
        out.collect(new Tuple2<String, Integer>(hashtag.get("text").asText(), 1));
      }
    }
  }
}
