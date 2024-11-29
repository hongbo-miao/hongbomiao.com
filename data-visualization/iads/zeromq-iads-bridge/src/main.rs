use chrono::{DateTime, Datelike, TimeZone, Utc};
use prost::Message;
use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use zeromq::{Socket, SocketRecv, SubSocket};

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

#[derive(Debug)]
struct Config {
    zeromq_host: String,
    zeromq_port: u16,
    iads_port: u16,
    iads_exe_path: PathBuf,
    iads_wizard_file_name: String,
    iads_parameter_definition_file_name: String,
    iads_config_file_path: PathBuf,
    iads_output_dir: PathBuf,
    temp_dir: TempDir,
}

impl Config {
    fn load() -> Self {
        // Load the appropriate .env file
        #[cfg(debug_assertions)]
        dotenvy::from_filename(".env.development").ok();
        #[cfg(not(debug_assertions))]
        dotenvy::from_filename(".env.production").ok();

        Self {
            zeromq_host: env::var("ZEROMQ_HOST").expect("ZEROMQ_HOST must be set in environment"),
            zeromq_port: env::var("ZEROMQ_PORT")
                .expect("ZEROMQ_PORT must be set in environment")
                .parse()
                .expect("ZEROMQ_PORT must be a valid port number"),
            iads_port: env::var("IADS_PORT")
                .expect("IADS_PORT must be set in environment")
                .parse()
                .expect("IADS_PORT must be a valid port number"),
            iads_exe_path: PathBuf::from(
                env::var("IADS_EXE_PATH").expect("IADS_EXE_PATH must be set in environment"),
            ),
            iads_wizard_file_name: env::var("IADS_WIZARD_FILE_NAME")
                .expect("IADS_WIZARD_FILE_NAME must be set in environment"),
            iads_parameter_definition_file_name: env::var("IADS_PARAMETER_DEFINITION_FILE_NAME")
                .expect("IADS_PARAMETER_DEFINITION_FILE_NAME must be set in environment"),
            iads_config_file_path: PathBuf::from(
                env::var("IADS_CONFIG_FILE_PATH")
                    .expect("IADS_CONFIG_FILE_PATH must be set in environment"),
            ),
            iads_output_dir: PathBuf::from(
                env::var("IADS_OUTPUT_DIR").expect("IADS_OUTPUT_DIR must be set in environment"),
            ),
            temp_dir: TempDir::new().expect("Failed to create temp directory"),
        }
    }

    fn get_zeromq_address(&self) -> String {
        format!("tcp://{}:{}", self.zeromq_host, self.zeromq_port)
    }
}

async fn generate_parameter_definition(
    signals: &Signals,
    setup_file: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(setup_file).await?;
    let frequency = signals.frequency_hz as f32;

    // Write IRIG timestamp
    // Code 1 for 32-bit unsigned integer
    file.write_all(
        format!(
            "1 IrigTimestampUpperWord {:.1} 1 SystemParamType = MajorTime\n",
            frequency
        )
        .as_bytes(),
    )
    .await?;
    file.write_all(
        format!(
            "2 IrigTimestampLowerWord {:.1} 1 SystemParamType = MinorTime\n",
            frequency
        )
        .as_bytes(),
    )
    .await?;

    // Write epoch unix timestamp
    // Code 4 for 64-bit unsigned integer
    file.write_all(format!("3 EpochUnixTimestampNs {:.1} 4\n", frequency).as_bytes())
        .await?;

    // Write signal parameters
    for (i, signal) in signals.signals.iter().enumerate() {
        // Code 2 for 32-bit single precision floating point
        file.write_all(format!("{} {} {:.1} 2\n", i + 4, signal.name, frequency).as_bytes())
            .await?;
    }
    Ok(())
}

async fn wait_for_first_message(get_zeromq_address: &str) -> Result<Signals, Box<dyn Error>> {
    let mut subscriber = SubSocket::new();
    subscriber.connect(get_zeromq_address).await?;
    subscriber.subscribe("").await?;

    let msg = subscriber.recv().await?;
    let default_bytes = prost::bytes::Bytes::new();
    let bytes = msg.get(0).unwrap_or(&default_bytes);

    let signals = Signals::decode(&bytes[..])?;
    Ok(signals)
}

fn calculate_packet_size_byte(signal_number: i32) -> i32 {
    let signal_pair_number = (signal_number + 1) / 2;
    IADS_HEADER_SIZE_BYTE
        + IADS_TIME_PAIR_SIZE_BYTE
        + 12
        + (signal_pair_number * IADS_SIGNAL_PAIR_SIZE_BYTE)
}

fn convert_epoch_to_iads_time(epoch_ns: i64) -> i64 {
    let now = Utc::now();
    let year_start_ns = Utc
        .with_ymd_and_hms(now.year(), 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap();
    let datetime = DateTime::<Utc>::from_timestamp_nanos(epoch_ns);
    let current_ns = datetime.timestamp_nanos_opt().unwrap();
    current_ns - year_start_ns
}

async fn process_zeromq_data(
    mut iads_stream: TcpStream,
    config: &Config,
) -> Result<(), Box<dyn Error>> {
    let mut subscriber = SubSocket::new();
    subscriber.connect(&config.get_zeromq_address()).await?;
    subscriber.subscribe("").await?;

    let mut packet_counter = 0i32;
    let mut buffer = Vec::new();
    let mut last_iads_time = 0i64;

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
        if last_iads_time != 0 {
            let interval = iads_time - last_iads_time;
            println!(
                "Epoch Time: {}, IADS Time: {}, Interval: {} ns ({} ms)",
                signals.timestamp_ns,
                iads_time,
                interval,
                interval / 1_000_000
            );
        }
        last_iads_time = iads_time;

        let time_high = ((iads_time >> 32) & 0xFFFFFFFF) as u32;
        let time_low = (iads_time & 0xFFFFFFFF) as u32;
        buffer[offset..offset + 4].copy_from_slice(&time_high.to_le_bytes());
        offset += 4;
        buffer[offset..offset + 4].copy_from_slice(&time_low.to_le_bytes());
        offset += 4;

        // Write EpochUnixTimestampNs as 64-bit value
        let tag1 = 3u16;
        buffer[offset..offset + 2].copy_from_slice(&tag1.to_le_bytes());
        offset += 2;
        buffer[offset..offset + 2].copy_from_slice(&0u16.to_le_bytes()); // Padding
        offset += 2;
        buffer[offset..offset + 8].copy_from_slice(&signals.timestamp_ns.to_le_bytes());
        offset += 8;

        // Write signal data in pairs
        for (i, signals_chunk) in signals.signals.chunks(2).enumerate() {
            // Write tags for pair
            let tag2 = (4 + i * 2) as u16;
            let tag3 = (5 + i * 2) as u16;
            buffer[offset..offset + 2].copy_from_slice(&tag2.to_le_bytes());
            offset += 2;
            buffer[offset..offset + 2].copy_from_slice(&tag3.to_le_bytes());
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

async fn update_wizard_file(
    wizard_file_path: &Path,
    iads_config_file_path: &Path,
    setup_file: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let wizard_file_content = fs::read_to_string("iads.WizardFile.template").await?;
    let updated_content = wizard_file_content
        .replace(
            "{IADS_CONFIG_FILE_PATH}",
            iads_config_file_path.to_str().unwrap_or(""),
        )
        .replace(
            "{IADS_PARAMETER_DEFINITION_PATH}",
            setup_file.to_str().unwrap_or(""),
        )
        .replace("{IADS_OUTPUT_DIR}", output_dir.to_str().unwrap_or(""));

    fs::write(wizard_file_path, updated_content).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::load();
    println!("{:#?}", config);

    // Wait for first message and generate parameter definition file
    println!("Waiting for first ZeroMQ message to generate parameter definition file...");
    let first_signals = wait_for_first_message(&config.get_zeromq_address()).await?;

    let iads_parameter_definition_path = config
        .temp_dir
        .path()
        .join(&config.iads_parameter_definition_file_name);
    generate_parameter_definition(&first_signals, &iads_parameter_definition_path).await?;
    println!("Parameter definition file generated successfully");

    // Update wizard file
    let wizard_file_path = config.temp_dir.path().join(&config.iads_wizard_file_name);
    update_wizard_file(
        &wizard_file_path,
        &config.iads_config_file_path,
        &iads_parameter_definition_path,
        &config.iads_output_dir,
    )
    .await?;
    println!(
        "WizardFile updated successfully with config path: {}",
        config.iads_config_file_path.display()
    );

    // Start IADS
    Command::new(&config.iads_exe_path)
        .args(["/rtstation", "Custom", wizard_file_path.to_str().unwrap()])
        .spawn()
        .expect("Failed to start IADS Client Workstation");

    let listener = TcpListener::bind(("0.0.0.0", config.iads_port)).await?;
    println!("Listening for IADS connection on port {}", config.iads_port);
    println!("Subscribing to ZeroMQ at {}", config.get_zeromq_address());

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                println!("IADS client connected");
                if let Err(e) = process_zeromq_data(stream, &config).await {
                    println!("Error processing data: {}", e);
                }
                println!("IADS client disconnected");
            }
            Err(e) => println!("Accept failed: {}", e),
        }
    }
}
