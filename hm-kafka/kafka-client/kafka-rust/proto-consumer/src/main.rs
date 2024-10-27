use rdkafka::client::ClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{CommitMode, Consumer, ConsumerContext, Rebalance, StreamConsumer};
use rdkafka::error::KafkaResult;
use rdkafka::message::Message;
use rdkafka::topic_partition_list::TopicPartitionList;
use schema_registry_converter::async_impl::easy_proto_raw::EasyProtoRawDecoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use serde_json::Value;
use std::env::args;

struct CustomContext;

impl ClientContext for CustomContext {}

impl ConsumerContext for CustomContext {
    fn pre_rebalance(&self, rebalance: &Rebalance) {
        println!("Pre rebalance {:?}", rebalance);
    }
    fn post_rebalance(&self, rebalance: &Rebalance) {
        println!("Post rebalance {:?}", rebalance);
    }
    fn commit_callback(&self, result: KafkaResult<()>, _offsets: &TopicPartitionList) {
        println!("Committing offsets: {:?}", result);
    }
}

fn create_consumer(bootstrap_server: &str, group_id: &str) -> StreamConsumer<CustomContext> {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .set("group.id", group_id)
        .set("enable.auto.commit", "true")
        .set("auto.offset.reset", "earliest")
        .set("session.timeout.ms", "6000")
        .create_with_context(CustomContext)
        .expect("Failed to create consumer")
}

#[tokio::main]
async fn main() {
    println!("Starting motor data Consumer...");
    let bootstrap_server = args()
        .nth(1)
        .unwrap_or_else(|| "localhost:9092".to_string());
    let consumer = create_consumer(&bootstrap_server, "proto-consumer-consumer-group");
    let schema_registry_url = "https://confluent-schema-registry.internal.hongbomiao.com";
    let sr_settings = SrSettings::new(schema_registry_url.to_string());
    let decoder = EasyProtoRawDecoder::new(sr_settings);
    let topic = "production.iot.motor.proto";

    consumer
        .subscribe(&[topic])
        .expect("Failed to subscribe to topic");
    println!("Subscribed to topic: {}", topic);

    loop {
        match consumer.recv().await {
            Ok(msg) => {
                match decoder.decode(msg.payload()).await {
                    Ok(decoded) => match serde_json::from_slice::<Value>(&decoded.unwrap().bytes) {
                        Ok(motor_data) => {
                            println!("Received data: {:?}", motor_data);
                        }
                        Err(e) => eprintln!("Failed to deserialize motor data: {}", e),
                    },
                    Err(e) => eprintln!("Error decoding message: {}", e),
                }
                consumer.commit_message(&msg, CommitMode::Async).unwrap();
            }
            Err(e) => eprintln!("Error while receiving message: {}", e),
        }
    }
}
