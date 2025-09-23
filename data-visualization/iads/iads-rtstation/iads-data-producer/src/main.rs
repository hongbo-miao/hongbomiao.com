#![deny(dead_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

use chrono::{Datelike, TimeZone, Utc};
use rand::Rng;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::time::{Duration, sleep};

// Constants for packet structure
const HEADER_SIZE_BYTE: i32 = 32;
const TAG_SIZE_BYTE: i32 = 2;
const VALUE_SIZE_BYTE: i32 = 4;
const TIME_PAIR_SIZE_BYTE: i32 = (TAG_SIZE_BYTE * 2) + (VALUE_SIZE_BYTE * 2); // 2 tags + 2 values
const SIGNAL_PAIR_SIZE_BYTE: i32 = (TAG_SIZE_BYTE * 2) + (VALUE_SIZE_BYTE * 2); // 2 tags + 2 values

// Configuration signals
struct StreamConfig {
    signal_number: i32,
    rate_hz: f64,
    port: u16,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            signal_number: 4,
            rate_hz: 50.0,
            port: 49000,
        }
    }
}

fn calculate_packet_size_byte(signal_number: i32) -> i32 {
    let signal_pair_number = (signal_number + 1) / 2;
    HEADER_SIZE_BYTE + TIME_PAIR_SIZE_BYTE + (signal_pair_number * SIGNAL_PAIR_SIZE_BYTE)
}

fn get_year_start_ns() -> i64 {
    let now = Utc::now();
    Utc.with_ymd_and_hms(now.year(), 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap()
}

fn get_iads_time() -> i64 {
    let year_start_ns = get_year_start_ns();
    let current_ns = Utc::now().timestamp_nanos_opt().unwrap();
    current_ns - year_start_ns
}

async fn send_data_stream(
    mut stream: tokio::net::TcpStream,
    config: &StreamConfig,
) -> std::io::Result<()> {
    let packet_size_byte = calculate_packet_size_byte(config.signal_number);
    let mut rng = rand::rng();

    // Calculate the interval in milliseconds from Hz
    let interval_ms = (1000.0 / config.rate_hz) as u64;
    let mut packet_counter = 0i32;
    let mut buffer = vec![0u8; packet_size_byte as usize];

    // Handshake
    stream.write_all(&[1u8]).await?;
    stream.write_all(&100i32.to_le_bytes()).await?;

    loop {
        let mut offset = 0usize;

        // Header
        buffer[offset..offset + VALUE_SIZE_BYTE as usize]
            .copy_from_slice(&(packet_size_byte - VALUE_SIZE_BYTE).to_le_bytes());
        buffer[offset + VALUE_SIZE_BYTE as usize..offset + VALUE_SIZE_BYTE as usize * 2]
            .copy_from_slice(&packet_counter.to_le_bytes());
        buffer[offset + VALUE_SIZE_BYTE as usize * 2..offset + VALUE_SIZE_BYTE as usize * 3]
            .copy_from_slice(&packet_counter.to_le_bytes());
        for i in 3..8 {
            buffer[offset + VALUE_SIZE_BYTE as usize * i
                ..offset + VALUE_SIZE_BYTE as usize * (i + 1)]
                .copy_from_slice(&0i32.to_le_bytes());
        }
        offset += HEADER_SIZE_BYTE as usize;

        // Time tags
        buffer[offset..offset + TAG_SIZE_BYTE as usize].copy_from_slice(&1u16.to_le_bytes());
        offset += TAG_SIZE_BYTE as usize;
        buffer[offset..offset + TAG_SIZE_BYTE as usize].copy_from_slice(&2u16.to_le_bytes());
        offset += TAG_SIZE_BYTE as usize;

        // Time value (nanoseconds since start of year)
        let iads_time = get_iads_time();
        let time_high = ((iads_time >> 32) & 0xFFFFFFFF) as u32;
        let time_low = (iads_time & 0xFFFFFFFF) as u32;

        buffer[offset..offset + VALUE_SIZE_BYTE as usize].copy_from_slice(&time_high.to_le_bytes());
        offset += VALUE_SIZE_BYTE as usize;
        buffer[offset..offset + VALUE_SIZE_BYTE as usize].copy_from_slice(&time_low.to_le_bytes());
        offset += VALUE_SIZE_BYTE as usize;

        // Signals with random values
        for n in (0..config.signal_number).step_by(2) {
            // Tags
            buffer[offset..offset + TAG_SIZE_BYTE as usize]
                .copy_from_slice(&((3 + n) as u16).to_le_bytes());
            offset += TAG_SIZE_BYTE as usize;
            buffer[offset..offset + TAG_SIZE_BYTE as usize]
                .copy_from_slice(&((4 + n) as u16).to_le_bytes());
            offset += TAG_SIZE_BYTE as usize;

            // Random values
            let value1: f32 = rng.random_range(0.0..100.0);
            let value2: f32 = rng.random_range(0.0..100.0);

            buffer[offset..offset + VALUE_SIZE_BYTE as usize]
                .copy_from_slice(&value1.to_le_bytes());
            offset += VALUE_SIZE_BYTE as usize;
            buffer[offset..offset + VALUE_SIZE_BYTE as usize]
                .copy_from_slice(&value2.to_le_bytes());
            offset += VALUE_SIZE_BYTE as usize;
        }

        if let Err(e) = stream.write_all(&buffer).await {
            println!("Send error: {}", e);
            return Ok(());
        }

        packet_counter += 1;
        sleep(Duration::from_millis(interval_ms)).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = StreamConfig::default();
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    println!("Listening on port {} at {} Hz", config.port, config.rate_hz);

    loop {
        println!("Waiting for client connection...");
        match listener.accept().await {
            Ok((stream, _)) => {
                println!("Client connected");
                send_data_stream(stream, &config).await?;
                println!("Client disconnected");
            }
            Err(e) => println!("Accept failed: {}", e),
        }
    }
}
