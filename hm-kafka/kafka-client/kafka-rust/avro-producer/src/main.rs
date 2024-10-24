use chrono::Utc;
use rand::Rng;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rdkafka::ClientConfig;
use schema_registry_converter::async_impl::easy_avro::EasyAvroEncoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use schema_registry_converter::schema_registry_common::SubjectNameStrategy;
use serde::Serialize;
use std::env::args;
use std::time::Duration;
use tokio::time;

#[derive(Serialize)]
struct SensorData {
    motor_id: String,
    timestamp: String,
    temperature: f64,
    humidity: f64,
    battery: i32,
}

fn create_producer(bootstrap_server: &str) -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .set("queue.buffering.max.ms", "0")
        .set("compression.type", "zstd")
        .create()
        .expect("Failed to create producer")
}

fn generate_sensor_data(motor_id: &str) -> SensorData {
    let mut rng = rand::thread_rng();
    SensorData {
        motor_id: motor_id.to_string(),
        timestamp: Utc::now().to_rfc3339(),
        temperature: rng.gen_range(18.0..28.0), // Temperature in Celsius
        humidity: rng.gen_range(30.0..70.0),    // Humidity percentage
        battery: rng.gen_range(0..101),         // Battery percentage (0-100)
    }
}

#[tokio::main]
async fn main() {
    println!("Starting IoT Data Generator...");

    // Get bootstrap server from args or use default
    let bootstrap_server = args()
        .nth(1)
        .unwrap_or_else(|| "localhost:9092".to_string());
    let producer = create_producer(&bootstrap_server);

    // Initialize EasyAvroEncoder
    let schema_registry_url = "https://confluent-schema-registry.internal.hongbomiao.com";
    let sr_settings = SrSettings::new(schema_registry_url.to_string());
    let encoder = EasyAvroEncoder::new(sr_settings);

    // Simulate 3 different IoT motors
    let motor_ids = vec!["motor_001", "motor_002", "motor_003"];
    let topic = "production.iot.motor.avro";

    println!("Sending data to Kafka topic: {}", topic);
    println!("Press Ctrl+C to stop");

    // Create an interval for sending data every 2 seconds
    // let mut interval = time::interval(Duration::from_secs(2));
    let mut interval = time::interval(Duration::from_nanos(10));

    loop {
        interval.tick().await;

        for motor_id in &motor_ids {
            let sensor_data = generate_sensor_data(motor_id);

            // Convert data to Avro bytes
            let avro_payload = encoder
                .encode_struct(
                    &sensor_data,
                    &SubjectNameStrategy::TopicNameStrategy(topic.to_string(), false),
                )
                .await
                .expect("Failed to encode sensor data to Avro");

            match producer
                .send(
                    FutureRecord::to(topic)
                        .payload(&avro_payload)
                        .key(motor_id.as_bytes()),
                    Timeout::After(Duration::from_secs(1)),
                )
                .await
            {
                Ok(_) => {}
                // Ok((partition, offset)) => {
                //     println!(
                //         "Sent data for motor {} to partition {} at offset {}",
                //         motor_id, partition, offset
                //     );
                // }
                Err((err, _)) => {
                    eprintln!("Failed to send data for motor {}: {}", motor_id, err);
                }
            }
        }
    }
}
