#![forbid(unsafe_code)]

use prost::Message;
use socket2::{Socket, Domain, Type};
use std::error::Error;
use std::net::SocketAddr;
use tokio::net::UdpSocket;

pub mod iot {
    include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
}

use iot::Motor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Create a standard UDP socket using socket2
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, None)?;
    socket.set_reuse_address(true)?;

    // Increase the buffer size to 20 MiB
    socket.set_recv_buffer_size(20 * 1024 * 1024)?;

    // Bind the socket to the address
    let addr: SocketAddr = "127.0.0.1:50537".parse()?;
    socket.bind(&addr.into())?;

    // Convert socket2::Socket into tokio::net::UdpSocket
    let socket = UdpSocket::from_std(socket.into())?;
    println!("Listening on 127.0.0.1:50537");

    let mut buffer = vec![0u8; 1024]; // Buffer to hold received data
    let mut message_count = 0; // Counter for received messages

    loop {
        let (amt, _src) = socket.recv_from(&mut buffer).await?;
        match Motor::decode(&buffer[..amt]) {
            Ok(_motor) => {
                message_count += 1;
                if message_count % 10000 == 0 {
                    println!("Total messages received: {}", message_count);
                }
            }
            Err(e) => {
                eprintln!("Failed to decode protobuf message: {}", e);
            }
        }
    }
}
