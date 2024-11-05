use prost::Message;
use std::error::Error;
use tokio::net::UdpSocket;

pub mod iot {
    include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
}

use iot::Motor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let socket = UdpSocket::bind("127.0.0.1:50537").await?;
    println!("Listening on 127.0.0.1:50537");

    let mut buffer = vec![0u8; 1024]; // Buffer to hold received data

    loop {
        let (amt, src) = socket.recv_from(&mut buffer).await?;
        match Motor::decode(&buffer[..amt]) {
            Ok(motor) => {
                println!("Received from {}: {:?}", src, motor);
            }
            Err(e) => {
                eprintln!("Failed to decode protobuf message: {}", e);
            }
        }
    }
}
