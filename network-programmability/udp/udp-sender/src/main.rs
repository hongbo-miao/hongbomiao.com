use chrono::Utc;
use rand::Rng;
use serde::Serialize;
use std::error::Error;
use tokio::net::UdpSocket;

#[derive(Serialize)]
struct Motor {
    id: Option<String>,
    timestamp: Option<i64>,
    temperature1: Option<f64>,
    temperature2: Option<f64>,
    temperature3: Option<f64>,
    temperature4: Option<f64>,
    temperature5: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let socket = UdpSocket::bind("0.0.0.0:0").await?; // Bind to any available address and port
    let receiver_addr = "127.0.0.1:50537";

    let mut rng = rand::thread_rng();
    let motor_ids = ["motor_001", "motor_002", "motor_003"];

    for _ in 0..1_000_000 {
        let motor_id = motor_ids[rng.gen_range(0..motor_ids.len())];
        let temperature = Some(rng.gen_range(10.0..100.0));
        let data = Motor {
            id: Some(motor_id.to_string()),
            timestamp: Utc::now().timestamp_nanos_opt(),
            temperature1: temperature,
            temperature2: temperature,
            temperature3: temperature,
            temperature4: temperature,
            temperature5: temperature,
        };
        let json_data = serde_json::to_string(&data)?;
        socket.send_to(json_data.as_bytes(), receiver_addr).await?; // Send the data asynchronously
    }

    println!("Sent 1 million messages.");
    Ok(())
}
