package com.hongbomiao;

import java.util.Properties;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
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
    DataStream<JsonNode> jsonStream =
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
                    return value.has("text") && !value.has("retweeted_status");
                  }
                });

    jsonStream.print();
    env.execute("Read Tweets");
  }
}
