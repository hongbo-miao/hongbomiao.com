#![forbid(unsafe_code)]

use chrono::Utc;
use prost::Message;
use rand::Rng;
use tokio::net::UdpSocket;
use std::time::Instant;

pub mod iot {
    include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
}
use iot::Motor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let message_count = 1_000_000;
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    let receiver_addr = "127.0.0.1:50537";
    let mut rng = rand::rng();
    let motor_ids = ["motor_001", "motor_002", "motor_003"];

    for _ in 0..message_count {
        let motor_id = motor_ids[rng.random_range(0..motor_ids.len())];
        let temperature = Some(rng.random_range(10.0..100.0));
        let motor = Motor {
            id: Some(motor_id.to_string()),
            timestamp_ns: Utc::now().timestamp_nanos_opt(),
            temperature1: temperature,
            temperature2: temperature,
            temperature3: temperature,
            temperature4: temperature,
            temperature5: temperature,
        };
        let mut buf = Vec::new();
        motor.encode(&mut buf)?;
        socket.send_to(&buf, receiver_addr).await?;
    }

    let duration = start_time.elapsed();
    let seconds = duration.as_secs_f64();
    let messages_per_second = message_count as f64 / seconds;

    println!("Sent 10 million messages in {:.2} seconds", seconds);
    println!("Performance: {:.2} messages/second", messages_per_second);

    Ok(())
}
