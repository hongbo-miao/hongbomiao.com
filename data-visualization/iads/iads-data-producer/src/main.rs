use rand::Rng;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::time::{sleep, Duration};

// Constants for packet structure
const HEADER_SIZE: i32 = 32;
const TAG_SIZE: i32 = 2;
const VALUE_SIZE: i32 = 4;
const TIME_PAIR_SIZE: i32 = (TAG_SIZE * 2) + (VALUE_SIZE * 2); // 2 tags + 2 values
const SIGNAL_PAIR_SIZE: i32 = (TAG_SIZE * 2) + (VALUE_SIZE * 2); // 2 tags + 2 values

// Configuration signals
struct StreamConfig {
    num_signals: i32,
    rate_hz: f64,
    port: u16,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            num_signals: 4,
            rate_hz: 50.0,
            port: 49000,
        }
    }
}

fn calculate_packet_size(num_signals: i32) -> i32 {
    let signal_pairs = (num_signals + 1) / 2;
    HEADER_SIZE + TIME_PAIR_SIZE + (signal_pairs * SIGNAL_PAIR_SIZE)
}

async fn send_data_stream(
    mut stream: tokio::net::TcpStream,
    config: &StreamConfig,
) -> std::io::Result<()> {
    let packet_size = calculate_packet_size(config.num_signals);
    let mut rng = rand::thread_rng();

    // Calculate the interval in milliseconds from Hz
    let interval_ms = (1000.0 / config.rate_hz) as u64;
    // Calculate time increment in nanoseconds from Hz
    let time_increment = (1.0 / config.rate_hz * 1_000_000_000.0) as i64;

    let mut time = 0i64;
    let mut packet_counter = 0i32;
    let mut buffer = vec![0u8; packet_size as usize];

    // Handshake
    stream.write_all(&[1u8]).await?;
    stream.write_all(&100i32.to_le_bytes()).await?;

    loop {
        let mut offset = 0usize;

        // Header
        buffer[offset..offset + VALUE_SIZE as usize]
            .copy_from_slice(&(packet_size - VALUE_SIZE).to_le_bytes());
        buffer[offset + VALUE_SIZE as usize..offset + VALUE_SIZE as usize * 2]
            .copy_from_slice(&packet_counter.to_le_bytes());
        buffer[offset + VALUE_SIZE as usize * 2..offset + VALUE_SIZE as usize * 3]
            .copy_from_slice(&packet_counter.to_le_bytes());
        for i in 3..8 {
            buffer[offset + VALUE_SIZE as usize * i..offset + VALUE_SIZE as usize * (i + 1)]
                .copy_from_slice(&0i32.to_le_bytes());
        }
        offset += HEADER_SIZE as usize;

        // Time tags and values
        for tag in [1u16, 2u16] {
            buffer[offset..offset + TAG_SIZE as usize].copy_from_slice(&tag.to_le_bytes());
            offset += TAG_SIZE as usize;
        }

        let time_high = (time >> 32) as u32;
        let time_low = (time & 0xFFFFFFFF) as u32;
        buffer[offset..offset + VALUE_SIZE as usize].copy_from_slice(&time_high.to_le_bytes());
        offset += VALUE_SIZE as usize;
        buffer[offset..offset + VALUE_SIZE as usize].copy_from_slice(&time_low.to_le_bytes());
        offset += VALUE_SIZE as usize;

        // Signals with random values
        for n in (0..config.num_signals).step_by(2) {
            // Tags
            buffer[offset..offset + TAG_SIZE as usize]
                .copy_from_slice(&((3 + n) as u16).to_le_bytes());
            offset += TAG_SIZE as usize;
            buffer[offset..offset + TAG_SIZE as usize]
                .copy_from_slice(&((4 + n) as u16).to_le_bytes());
            offset += TAG_SIZE as usize;

            // Random values
            let value1: f32 = rng.gen_range(0.0..100.0);
            let value2: f32 = rng.gen_range(0.0..100.0);

            buffer[offset..offset + VALUE_SIZE as usize].copy_from_slice(&value1.to_le_bytes());
            offset += VALUE_SIZE as usize;
            buffer[offset..offset + VALUE_SIZE as usize].copy_from_slice(&value2.to_le_bytes());
            offset += VALUE_SIZE as usize;
        }

        if let Err(e) = stream.write_all(&buffer).await {
            println!("Send error: {}", e);
            return Ok(());
        }

        packet_counter += 1;
        time += time_increment;
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
