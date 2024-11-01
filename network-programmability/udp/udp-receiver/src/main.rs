use serde_json::Value;
use std::error::Error;
use tokio::net::UdpSocket;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let socket = UdpSocket::bind("127.0.0.1:50537").await?;
    println!("Listening on 127.0.0.1:50537");

    let mut buffer = vec![0u8; 1024]; // Buffer to hold received data

    loop {
        let (amt, src) = socket.recv_from(&mut buffer).await?;
        let received = String::from_utf8_lossy(&buffer[..amt]);
        match serde_json::from_str::<Value>(&received) {
            Ok(json_data) => {
                println!("Received from {}: {}", src, json_data);
            }
            Err(e) => {
                eprintln!("Failed to parse JSON: {}", e);
            }
        }
    }
}
