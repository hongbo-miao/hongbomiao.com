#![forbid(unsafe_code)]

use prost::Message;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::message::Message as KafkaMessage;
use schema_registry_converter::async_impl::easy_proto_raw::EasyProtoRawDecoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use std::env::args;

pub mod iot {
    include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
}
use iot::Motor;

fn create_consumer(bootstrap_server: &str, group_id: &str) -> StreamConsumer {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .set("group.id", group_id)
        .set("enable.auto.commit", "true")
        .set("auto.offset.reset", "earliest")
        .set("session.timeout.ms", "6000")
        .create()
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
                if let Some(payload) = msg.payload() {
                    match decoder.decode(Some(payload)).await {
                        Ok(Some(decoded)) => {
                            match Motor::decode(&*decoded.bytes) {
                                Ok(motor) => {
                                    println!("Received motor data:");
                                    println!("  id: {:?}", motor.id);
                                    println!("  timestamp_ns: {:?}", motor.timestamp_ns);
                                    println!("  temperature1: {:?}", motor.temperature1);
                                    println!("  temperature2: {:?}", motor.temperature2);
                                    println!("  temperature3: {:?}", motor.temperature3);
                                    println!("  temperature4: {:?}", motor.temperature4);
                                    println!("  temperature5: {:?}", motor.temperature5);
                                }
                                Err(e) => eprintln!("Failed to decode motor data: {}", e),
                            }
                        }
                        Ok(None) => eprintln!("No data decoded"),
                        Err(e) => eprintln!("Error decoding message: {}", e),
                    }
                }
                consumer.commit_message(&msg, CommitMode::Async).unwrap();
            }
            Err(e) => eprintln!("Error while receiving message: {}", e),
        }
    }
}
