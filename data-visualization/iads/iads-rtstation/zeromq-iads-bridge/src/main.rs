use axum::http::StatusCode;
use axum::response::IntoResponse;
use chrono::{DateTime, Datelike, TimeZone, Utc};
use prost::Message;
use serde::Serialize;
use std::env;
use std::error::Error;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::RwLock;
use windows::core::{HSTRING, PCWSTR};
use windows::Win32::System::Com::{CLSIDFromProgID, CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED, IDispatch, DISPPARAMS, DISPATCH_METHOD};
use windows::Win32::System::Variant::VARIANT;
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

#[derive(Debug, Clone, Serialize)]
struct IadsStatus {
    is_iads_connected: bool,
    iads_last_updated_timestamp_ns: i64,
    iads_processed_message_count: i32,
}

type SharedIadsStatus = Arc<RwLock<IadsStatus>>;

#[derive(Debug)]
struct Config {
    zeromq_host: String,
    zeromq_port: u16,
    iads_port: u16,
    api_server_port: u16,
    iads_exe_path: PathBuf,
    iads_wizard_file_name: String,
    iads_parameter_definition_file_name: String,
    iads_config_file_path: PathBuf,
    iads_data_output_dir: PathBuf,
    temp_dir: TempDir,
}

impl Config {
    fn load() -> Self {
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
            iads_data_output_dir: PathBuf::from(
                env::var("IADS_DATA_OUTPUT_DIR")
                    .expect("IADS_DATA_OUTPUT_DIR must be set in environment"),
            ),
            api_server_port: env::var("API_SERVER_PORT")
                .expect("API_SERVER_PORT must be set in environment")
                .parse()
                .expect("API_SERVER_PORT must be a valid port number"),
            temp_dir: TempDir::new().expect("Failed to create temp directory"),
        }
    }

    fn get_zeromq_address(&self) -> String {
        format!("tcp://{}:{}", self.zeromq_host, self.zeromq_port)
    }
}

async fn get_iads_status(
    axum::extract::State(shared_iads_status): axum::extract::State<SharedIadsStatus>,
) -> axum::Json<IadsStatus> {
    let iads_status = shared_iads_status.read().await;
    axum::Json(iads_status.clone())
}

async fn stop_iads() -> Result<(), Box<dyn Error>> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let prog_id: HSTRING = "Iads.Application".into();
        let clsid = CLSIDFromProgID(PCWSTR::from_raw(prog_id.as_ptr()))?;
        let iads: IDispatch = CoCreateInstance(&clsid, None, CLSCTX_ALL)?;

        // Quit
        let method_name: HSTRING = "Quit".into();
        let mut disp_id = 0;
        let names = [PCWSTR::from_raw(method_name.as_ptr())];
        let result = iads.GetIDsOfNames(
            &windows::core::GUID::zeroed(),
            names.as_ptr(),
            1,
            0,
            &mut disp_id,
        );

        if result.is_ok() {
            let params = DISPPARAMS::default();
            let mut result: VARIANT = std::mem::zeroed();
            let _ = iads.Invoke(
                disp_id,
                &windows::core::GUID::zeroed(),
                0,
                DISPATCH_METHOD,
                &params,
                Some(&mut result),
                None,
                None,
            );
        }

        Ok(())
    }
}

async fn run_api_server(shared_iads_status: SharedIadsStatus, api_server_port: u16) {
    let app = axum::Router::new()
        .route("/status", axum::routing::get(get_iads_status))
        .route(
            "/stop",
            axum::routing::post(|| async {
                match stop_iads().await {
                    Ok(_) => (StatusCode::OK, "IADS stopped successfully").into_response(),
                    Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
                }
            }),
        )
        .with_state(shared_iads_status);

    let addr = SocketAddr::from(([0, 0, 0, 0], api_server_port));
    println!("API server listening on {}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
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

    // Write unix timestamp
    // Code 4 for 64-bit unsigned integer
    file.write_all(format!("3 UnixTimestampNs {:.1} 4\n", frequency).as_bytes())
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

fn convert_unix_to_iads_time(unix_timestamp_ns: i64) -> i64 {
    let now = Utc::now();
    let year_start_ns = Utc
        .with_ymd_and_hms(now.year(), 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap();
    let datetime = DateTime::<Utc>::from_timestamp_nanos(unix_timestamp_ns);
    let current_ns = datetime.timestamp_nanos_opt().unwrap();
    current_ns - year_start_ns
}

async fn process_zeromq_data(
    mut iads_stream: TcpStream,
    config: &Config,
    shared_iads_status: SharedIadsStatus,
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

        // Update IADS status
        {
            let mut iads_status = shared_iads_status.write().await;
            iads_status.iads_last_updated_timestamp_ns =
                Utc::now().timestamp_nanos_opt().unwrap_or_else(|| {
                    println!("Warning: Timestamp overflow occurred, using 0 as fallback");
                    0
                });
            iads_status.iads_processed_message_count = packet_counter;
        }

        let signal_count = signals.signals.len();

        // Print signal information for the first message or when count changes
        if packet_counter == 0 {
            println!("\nSignal Count: {}", signal_count);
            println!("Timestamp: {} ns", signals.timestamp_ns);
            println!("Signal Names:");
            for (i, signal) in signals.signals.iter().enumerate() {
                println!(" {}: {} = {}", i, signal.name, signal.value);
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
        let iads_time = convert_unix_to_iads_time(signals.timestamp_ns);

        // Print time interval information
        if last_iads_time != 0 {
            let interval = iads_time - last_iads_time;
            println!(
                "Unix Time: {}, IADS Time: {}, Interval: {} ns ({} ms)",
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

        // Write UnixTimestampNs as 64-bit value
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
    iads_port: u16,
    iads_wizard_file_path: &Path,
    iads_parameter_definition_file_path: &Path,
    iads_config_file_path: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let wizard_file_content = fs::read_to_string("iads.WizardFile.template").await?;
    let updated_content = wizard_file_content
        .replace("{IADS_PORT}", &iads_port.to_string())
        .replace(
            "{IADS_CONFIG_FILE_PATH}",
            iads_config_file_path.to_str().unwrap_or(""),
        )
        .replace(
            "{IADS_PARAMETER_DEFINITION_FILE_PATH}",
            iads_parameter_definition_file_path.to_str().unwrap_or(""),
        )
        .replace("{IADS_DATA_OUTPUT_DIR}", output_dir.to_str().unwrap_or(""));

    fs::write(iads_wizard_file_path, updated_content).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::load();
    println!("{:#?}", config);

    // Initialize shared status
    let shared_iads_status = Arc::new(RwLock::new(IadsStatus {
        is_iads_connected: false,
        iads_last_updated_timestamp_ns: 0,
        iads_processed_message_count: 0,
    }));

    // Start API server
    let api_shared_status = shared_iads_status.clone();
    tokio::spawn(async move {
        run_api_server(api_shared_status, config.api_server_port).await;
    });

    // Wait for first ZeroMQ message and generate parameter definition file
    println!("Waiting for first ZeroMQ message to generate parameter definition file...");
    let first_signals = wait_for_first_message(&config.get_zeromq_address()).await?;

    let iads_parameter_definition_file_path = config
        .temp_dir
        .path()
        .join(&config.iads_parameter_definition_file_name);
    generate_parameter_definition(&first_signals, &iads_parameter_definition_file_path).await?;
    println!("Parameter definition file generated successfully");

    // Update IADS wizard file
    let iads_wizard_file_path = config.temp_dir.path().join(&config.iads_wizard_file_name);
    update_wizard_file(
        config.iads_port,
        &iads_wizard_file_path,
        &iads_parameter_definition_file_path,
        &config.iads_config_file_path,
        &config.iads_data_output_dir,
    )
    .await?;
    println!(
        "WizardFile updated successfully with config path: {}",
        config.iads_config_file_path.display()
    );

    // Start IADS process
    let mut child = Command::new(&config.iads_exe_path)
        .args([
            "/rtstation",
            "Custom",
            iads_wizard_file_path.to_str().unwrap(),
        ])
        .spawn()
        .expect("Failed to start IADS");

    tokio::spawn(async move {
        if let Err(e) = child.wait().await {
            eprintln!("IADS process exited with error: {}", e);
        } else {
            println!("IADS process exited gracefully.");
        }
    });

    // Start the TCP listener
    let listener = TcpListener::bind(("0.0.0.0", config.iads_port)).await?;
    println!("Listening for IADS connection on port {}", config.iads_port);
    println!("Subscribing to ZeroMQ at {}", config.get_zeromq_address());

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                println!("IADS client connected");
                {
                    let mut iads_status = shared_iads_status.write().await;
                    iads_status.is_iads_connected = true;
                }

                if let Err(e) =
                    process_zeromq_data(stream, &config, shared_iads_status.clone()).await
                {
                    println!("Error processing data: {}", e);
                }

                {
                    let mut iads_status = shared_iads_status.write().await;
                    iads_status.is_iads_connected = false;
                }
                println!("IADS client disconnected...");
            }
            Err(e) => eprintln!("Error accepting connection: {}", e),
        }
    }
}
