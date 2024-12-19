use chrono::Utc;
use prost::Message;
use rand::Rng;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rdkafka::ClientConfig;
use schema_registry_converter::async_impl::easy_proto_raw::EasyProtoRawEncoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use schema_registry_converter::schema_registry_common::SubjectNameStrategy;
use std::env::args;
use std::io::Result;
use std::time::Duration;
use tokio::time;

pub mod iot {
    include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
}
use iot::Motor;

fn create_producer(bootstrap_server: &str) -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .set("batch.size", "20971520") // 20 MiB
        .set("linger.ms", "5")
        .create()
        .expect("Failed to create producer")
}

fn generate_motor_data(motor_id: &str) -> Motor {
    let mut rng = rand::rng();
    let temperature = rng.random_range(10.0..100.0);
    Motor {
        id: Some(motor_id.to_string()),
        timestamp_ns: Utc::now().timestamp_nanos_opt(),
        temperature1: Some(temperature),
        temperature2: Some(temperature),
        temperature3: Some(temperature),
        temperature4: Some(temperature),
        temperature5: Some(temperature),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting motor data generator...");

    let bootstrap_server = args()
        .nth(1)
        .unwrap_or_else(|| "localhost:9092".to_string());
    let producer = create_producer(&bootstrap_server);

    let schema_registry_url = "https://confluent-schema-registry.internal.hongbomiao.com";
    let sr_settings = SrSettings::new(schema_registry_url.to_string());
    let encoder = EasyProtoRawEncoder::new(sr_settings);
    let mut interval = time::interval(Duration::from_nanos(100));
    let mut rng = rand::rng();
    let topic = "production.iot.motor.proto";
    let motor_ids = ["motor_001", "motor_002", "motor_003"];

    println!("Sending data to Kafka topic: {}", topic);
    loop {
        interval.tick().await;

        let motor_id = motor_ids[rng.random_range(0..motor_ids.len())];
        let motor = generate_motor_data(motor_id);
        let mut buf = Vec::new();
        motor.encode(&mut buf)?;

        let proto_payload = encoder
            .encode(
                &buf,
                "production.iot.Motor",
                SubjectNameStrategy::TopicNameStrategy(topic.to_string(), false),
            )
            .await
            .expect("Failed to encode with schema registry");

        match producer
            .send(
                FutureRecord::to(topic)
                    .payload(&proto_payload)
                    .key(motor_id.as_bytes()),
                Timeout::After(Duration::from_secs(1)),
            )
            .await
        {
            Ok(_) => {}
            Err((err, _)) => {
                eprintln!("Failed to send data for motor {}: {}", motor_id, err);
            }
        }
    }
}
