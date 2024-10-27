use chrono::Utc;
use rand::Rng;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rdkafka::ClientConfig;
use schema_registry_converter::async_impl::easy_proto_raw::EasyProtoRawEncoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use schema_registry_converter::schema_registry_common::SubjectNameStrategy;
use serde_json::json;
use std::env::args;
use std::time::Duration;
use tokio::time;

fn create_producer(bootstrap_server: &str) -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .create()
        .expect("Failed to create producer")
}

fn generate_motor_data(motor_id: &str) -> serde_json::Value {
    let mut rng = rand::thread_rng();
    let temperature = rng.gen_range(10.0..100.0);
    json!({
        "motor_id": motor_id,
        "timestamp": Utc::now().to_rfc3339(),
        "temperature1": temperature,
        "temperature2": temperature,
        "temperature3": temperature,
        "temperature4": temperature,
        "temperature5": temperature
    })
}

#[tokio::main]
async fn main() {
    println!("Starting motor data Generator...");

    let bootstrap_server = args()
        .nth(1)
        .unwrap_or_else(|| "localhost:9092".to_string());
    let producer = create_producer(&bootstrap_server);

    let schema_registry_url = "https://confluent-schema-registry.internal.hongbomiao.com";
    let sr_settings = SrSettings::new(schema_registry_url.to_string());
    let encoder = EasyProtoRawEncoder::new(sr_settings);

    let motor_ids = vec!["motor_001", "motor_002", "motor_003"];
    let topic = "production.iot.motor.proto";

    println!("Sending data to Kafka topic: {}", topic);

    let mut interval = time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;

        for motor_id in &motor_ids {
            let sensor_data = generate_motor_data(motor_id);
            let json_bytes =
                serde_json::to_vec(&sensor_data).expect("Failed to serialize sensor data");
            let proto_payload = encoder
                .encode(
                    &json_bytes,
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
}
