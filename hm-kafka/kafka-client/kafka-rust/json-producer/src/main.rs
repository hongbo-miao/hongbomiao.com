use chrono::Utc;
use rand::Rng;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rdkafka::ClientConfig;
use serde::Serialize;
use std::env::args;
use std::time::Duration;
use tokio::time;

#[derive(Serialize)]
struct SensorData {
    device_id: String,
    timestamp: String,
    temperature: f64,
    humidity: f64,
    battery: i32,
}

fn create_producer(bootstrap_server: &str) -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .set("queue.buffering.max.ms", "0")
        .create()
        .expect("Failed to create producer")
}

fn generate_sensor_data(device_id: &str) -> SensorData {
    let mut rng = rand::thread_rng();

    SensorData {
        device_id: device_id.to_string(),
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

    // Simulate 3 different IoT devices
    let device_ids = vec!["device_001", "device_002", "device_003"];
    let topic = "production.iot.device.json";

    println!("Sending data to Kafka topic: {}", topic);
    println!("Press Ctrl+C to stop");

    // Create an interval for sending data every 2 seconds
    let mut interval = time::interval(Duration::from_secs(2));

    loop {
        interval.tick().await;

        for device_id in &device_ids {
            let sensor_data = generate_sensor_data(device_id);
            let payload =
                serde_json::to_string(&sensor_data).expect("Failed to serialize sensor data");

            match producer
                .send(
                    FutureRecord::to(topic)
                        .payload(&payload)
                        .key(device_id.as_bytes()), // Fixed: Convert string to bytes
                    Timeout::After(Duration::from_secs(1)),
                )
                .await
            {
                Ok((partition, offset)) => {
                    println!(
                        "Sent data for device {} to partition {} at offset {}: {}",
                        device_id, partition, offset, payload
                    );
                }
                Err((err, _)) => {
                    eprintln!("Failed to send data for device {}: {}", device_id, err);
                }
            }
        }
    }
}
