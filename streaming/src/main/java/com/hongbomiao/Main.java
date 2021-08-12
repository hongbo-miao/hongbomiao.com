package com.hongbomiao;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.util.Collector;

import java.util.Properties;

public class Main {
  public static void main(String[] args) throws Exception {
    final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

    Properties props = new Properties();
    props.setProperty(TwitterSource.CONSUMER_KEY, "");
    props.setProperty(TwitterSource.CONSUMER_SECRET, "");
    props.setProperty(TwitterSource.TOKEN, "");
    props.setProperty(TwitterSource.TOKEN_SECRET, "");

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
