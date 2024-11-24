use std::error::Error;
use std::fs::File;
use std::io::Write;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use prost::Message;
use zeromq::{Socket, SocketRecv, SubSocket};
use tokio::net::TcpListener;
use chrono::{DateTime, Datelike, TimeZone, Utc};

pub mod production {
    pub mod iot {
        include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
    }
}
use production::iot::Signals;

const IADS_HEADER_SIZE_BYTE: i32 = 32;
const IADS_TAG_SIZE_BYTE: i32 = 2;
const IADS_VALUE_SIZE_BYTE: i32 = 4;
const IADS_TIME_PAIR_SIZE_BYTE: i32 = (IADS_TAG_SIZE_BYTE * 2) + (IADS_VALUE_SIZE_BYTE * 2);
const IADS_SIGNAL_PAIR_SIZE_BYTE: i32 = (IADS_TAG_SIZE_BYTE * 2) + (IADS_VALUE_SIZE_BYTE * 2);

struct StreamConfig {
    port: u16,
    zmq_address: String,
    last_iads_time: i64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            port: 49000,
            zmq_address: "tcp://10.0.0.100:5555".to_string(),
            last_iads_time: 0,
        }
    }
}

fn generate_parameter_definition(signals: &Signals) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("iadsParameterDefinition.prn")?;
    let frequency = signals.frequency_hz as f32;

    // Write timestamp parameters
    writeln!(file, "1 TimestampUpperWord {:.1} 1 SystemParamType = MajorTime", frequency)?;
    writeln!(file, "2 TimestampLowerWord {:.1} 1 SystemParamType = MinorTime", frequency)?;

    // Write signal parameters
    for (i, signal) in signals.signals.iter().enumerate() {
        writeln!(file, "{} {} {:.1} 2",
                 i + 3,
                 signal.name,
                 frequency
        )?;
    }

    Ok(())
}

async fn wait_for_first_message(zmq_address: &str) -> Result<Signals, Box<dyn Error>> {
    let mut subscriber = SubSocket::new();
    subscriber.connect(zmq_address).await?;
    subscriber.subscribe("").await?;

    let msg = subscriber.recv().await?;
    let default_bytes = prost::bytes::Bytes::new();
    let bytes = msg.get(0).unwrap_or(&default_bytes);

    let signals = Signals::decode(&bytes[..])?;
    Ok(signals)
}

fn calculate_packet_size_byte(signal_number: i32) -> i32 {
    let signal_pair_number = (signal_number + 1) / 2;
    IADS_HEADER_SIZE_BYTE + IADS_TIME_PAIR_SIZE_BYTE + (signal_pair_number * IADS_SIGNAL_PAIR_SIZE_BYTE)
}

fn convert_epoch_to_iads_time(epoch_ns: i64) -> i64 {
    let now = Utc::now();
    let year_start_ns = Utc.with_ymd_and_hms(now.year(), 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap();
    let datetime = DateTime::<Utc>::from_timestamp_nanos(epoch_ns);
    let current_ns = datetime.timestamp_nanos_opt().unwrap();
    current_ns - year_start_ns
}

async fn process_zmq_data(
    mut iads_stream: TcpStream,
    config: &mut StreamConfig,
) -> Result<(), Box<dyn Error>> {
    let mut subscriber = SubSocket::new();
    subscriber.connect(&config.zmq_address).await?;
    subscriber.subscribe("").await?;

    let mut packet_counter = 0i32;
    let mut buffer = Vec::new();

    // Handshake with IADS
    iads_stream.write_all(&[1u8]).await?;
    iads_stream.write_all(&100i32.to_le_bytes()).await?;

    loop {
        let msg = subscriber.recv().await?;
        let default_bytes = prost::bytes::Bytes::new();
        let bytes = msg.get(0).unwrap_or(&default_bytes);

        // Deserialize prost message
        let signals = match Signals::decode(&bytes[..]) {
            Ok(signals) => signals,
            Err(e) => {
                println!("Failed to decode protobuf message: {}", e);
                continue;
            }
        };

        let signal_count = signals.signals.len();

        // Print signal information for the first message or when count changes
        if packet_counter == 0 {
            println!("\nSignal Count: {}", signal_count);
            println!("Timestamp: {} ns", signals.timestamp_ns);
            println!("Signal Names:");
            for (i, signal) in signals.signals.iter().enumerate() {
                println!("  {}: {} = {}", i, signal.name, signal.value);
            }
            println!("\nStarting data processing...\n");
        }

        let packet_size = calculate_packet_size_byte(signal_count as i32);
        buffer.resize(packet_size as usize, 0);

        let mut offset = 0usize;

        // Write IADS header
        buffer[offset..offset + 4].copy_from_slice(&(packet_size - 4).to_le_bytes());
        buffer[offset + 4..offset + 8].copy_from_slice(&packet_counter.to_le_bytes());
        buffer[offset + 8..offset + 12].copy_from_slice(&packet_counter.to_le_bytes());
        offset += IADS_HEADER_SIZE_BYTE as usize;

        // Write time tags
        for tag in [1u16, 2u16] {
            buffer[offset..offset + 2].copy_from_slice(&tag.to_le_bytes());
            offset += 2;
        }

        // Write time values
        let iads_time = convert_epoch_to_iads_time(signals.timestamp_ns);

        // Print time interval information
        if config.last_iads_time != 0 {
            let interval = iads_time - config.last_iads_time;
            println!("IADS Time: {}, Interval: {} ns ({} ms)",
                     iads_time,
                     interval,
                     interval / 1_000_000
            );
        }
        config.last_iads_time = iads_time;

        let time_high = ((iads_time >> 32) & 0xFFFFFFFF) as u32;
        let time_low = (iads_time & 0xFFFFFFFF) as u32;
        buffer[offset..offset + 4].copy_from_slice(&time_high.to_le_bytes());
        offset += 4;
        buffer[offset..offset + 4].copy_from_slice(&time_low.to_le_bytes());
        offset += 4;

        // Write signal data in pairs
        for (i, signals_chunk) in signals.signals.chunks(2).enumerate() {
            // Write tags for pair
            let tag1 = (3 + i * 2) as u16;
            let tag2 = (4 + i * 2) as u16;
            buffer[offset..offset + 2].copy_from_slice(&tag1.to_le_bytes());
            offset += 2;
            buffer[offset..offset + 2].copy_from_slice(&tag2.to_le_bytes());
            offset += 2;

            // Write first value
            buffer[offset..offset + 4].copy_from_slice(&signals_chunk[0].value.to_le_bytes());
            offset += 4;

            // Write second value (or 0 if this is an odd-numbered final chunk)
            let second_value = signals_chunk.get(1).map(|s| s.value).unwrap_or(0.0);
            buffer[offset..offset + 4].copy_from_slice(&second_value.to_le_bytes());
            offset += 4;
        }

        // Send to IADS
        if let Err(e) = iads_stream.write_all(&buffer).await {
            println!("IADS send error: {}", e);
            break;
        }

        packet_counter += 1;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut config = StreamConfig::default();

    // Wait for first message and generate parameter definition file
    println!("Waiting for first ZeroMQ message to generate parameter definition file...");
    let first_signals = wait_for_first_message(&config.zmq_address).await?;
    generate_parameter_definition(&first_signals)?;
    println!("Parameter definition file generated successfully");

    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    println!("Listening for IADS connection on port {}", config.port);
    println!("Subscribing to ZeroMQ at {}", config.zmq_address);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                println!("IADS client connected");
                if let Err(e) = process_zmq_data(stream, &mut config).await {
                    println!("Error processing data: {}", e);
                }
                println!("IADS client disconnected");
            }
            Err(e) => println!("Accept failed: {}", e),
        }
    }
}
