use chrono::Utc;
use rand::Rng;
use serde::Serialize;
use std::error::Error;
use tokio::net::UdpSocket;

#[derive(Serialize)]
struct MotorData {
    motor_id: Option<String>,
    timestamp: Option<String>,
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
        let temperature = rng.gen_range(10.0..100.0);
        let data = MotorData {
            motor_id: Some(motor_id.to_string()),
            timestamp: Some(Utc::now().to_rfc3339()),
            temperature1: Some(temperature),
            temperature2: Some(temperature),
            temperature3: Some(temperature),
            temperature4: Some(temperature),
            temperature5: Some(temperature),
        };
        let json_data = serde_json::to_string(&data)?;
        socket.send_to(json_data.as_bytes(), receiver_addr).await?; // Send the data asynchronously
    }

    println!("Sent 1 million messages.");
    Ok(())
}
