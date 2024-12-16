#![forbid(unsafe_code)]

use prost::Message;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rdkafka::ClientConfig;
use schema_registry_converter::async_impl::easy_proto_raw::EasyProtoRawEncoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use schema_registry_converter::schema_registry_common::SubjectNameStrategy;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use zeromq::{Socket, SocketRecv, SubSocket};
use std::env;

pub mod production {
    pub mod iot {
        include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
    }
}
use production::iot::Signals;

#[derive(Debug)]
struct Config {
    zeromq_host: String,
    zeromq_port: u16,
    kafka_bootstrap_servers: String,
    schema_registry_url: String,
    kafka_topic: String,
    channel_buffer_size: usize,
}

impl Config {
    fn load() -> Self {
        #[cfg(debug_assertions)]
        dotenvy::from_filename(".env.development").ok();
        #[cfg(not(debug_assertions))]
        dotenvy::from_filename(".env.production").ok();

        Self {
            zeromq_host: env::var("ZEROMQ_HOST")
                .expect("ZEROMQ_HOST must be set in environment"),
            zeromq_port: env::var("ZEROMQ_PORT")
                .expect("ZEROMQ_PORT must be set in environment")
                .parse()
                .expect("ZEROMQ_PORT must be a valid port number"),
            kafka_bootstrap_servers: env::var("KAFKA_BOOTSTRAP_SERVERS")
                .expect("KAFKA_BOOTSTRAP_SERVERS must be set in environment"),
            schema_registry_url: env::var("SCHEMA_REGISTRY_URL")
                .expect("SCHEMA_REGISTRY_URL must be set in environment"),
            kafka_topic: env::var("KAFKA_TOPIC")
                .expect("KAFKA_TOPIC must be set in environment"),
            channel_buffer_size: env::var("CHANNEL_BUFFER_SIZE")
                .expect("CHANNEL_BUFFER_SIZE must be set in environment")
                .parse()
                .expect("CHANNEL_BUFFER_SIZE must be a valid number"),
        }
    }

    fn get_zeromq_address(&self) -> String {
        format!("tcp://{}:{}", self.zeromq_host, self.zeromq_port)
    }
}

struct ProcessingContext {
    producer: FutureProducer,
    encoder: EasyProtoRawEncoder,
    topic: String,
    messages_sent: AtomicUsize,
    messages_received: AtomicUsize,
    last_count: AtomicUsize,
    last_time: Mutex<Instant>,
}

fn create_producer(bootstrap_server: &str) -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", bootstrap_server)
        .set("batch.size", "1048576") // 1 MiB
        .set("queue.buffering.max.messages", "100000")
        .set("queue.buffering.max.kbytes", "1048576") // 1GB max memory usage
        .set("linger.ms", "2")
        //.set("compression.type", "zstd")
        .create()
        .expect("Failed to create producer")
}

async fn process_message(
    context: Arc<ProcessingContext>,
    signals: Signals,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut buf = Vec::new();
    signals.encode(&mut buf)?;
    let proto_payload = context
        .encoder
        .encode(
            &buf,
            "production.iot.Signals",
            SubjectNameStrategy::TopicNameStrategy(context.topic.clone(), false),
        )
        .await
        .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    context
        .producer
        .send(
            FutureRecord::to(&context.topic)
                .payload(&proto_payload)
                .key(&signals.timestamp_ns.to_string()),
            Timeout::After(Duration::from_secs(1)),
        )
        .await
        .map_err(|(err, _)| Box::new(err) as Box<dyn Error + Send + Sync>)?;

    context.messages_sent.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::load();
    println!("{:#?}", config);

    println!("Starting ZeroMQ subscriber and Kafka producer...");

    const LOW_CAPACITY_THRESHOLD: usize = 10000; // 10% of default channel buffer size
    const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5);

    // Set up ZeroMQ subscriber
    let mut socket = SubSocket::new();
    socket.connect(&config.get_zeromq_address()).await?;
    socket.subscribe("").await?;
    println!("Connected to ZeroMQ publisher");

    // Set up Kafka producer and Schema Registry encoder
    let producer = create_producer(&config.kafka_bootstrap_servers);
    let sr_settings = SrSettings::new(config.schema_registry_url.clone());
    let encoder = EasyProtoRawEncoder::new(sr_settings);

    // Create processing context
    let context = Arc::new(ProcessingContext {
        producer,
        encoder,
        topic: config.kafka_topic.clone(),
        messages_sent: AtomicUsize::new(0),
        messages_received: AtomicUsize::new(0),
        last_count: AtomicUsize::new(0),
        last_time: Mutex::new(Instant::now()),
    });

    // Create channel for message processing
    let (tx, mut rx) = mpsc::channel::<Signals>(config.channel_buffer_size);

    // Spawn message processing task
    let processing_context = context.clone();
    tokio::spawn(async move {
        let mut last_message_time = Instant::now();
        let mut handles = Vec::new();
        loop {
            tokio::select! {
                Some(signals) = rx.recv() => {
                    let ctx = processing_context.clone();
                    last_message_time = Instant::now();
                    handles.push(tokio::spawn(async move {
                        if let Err(e) = process_message(ctx, signals).await {
                            eprintln!("Error processing message: {}", e);
                        }
                    }));
                    handles.retain(|handle| !handle.is_finished());
                    while handles.len() >= 16 {
                        futures::future::join_all(handles.drain(..8)).await;
                    }
                },
                _ = tokio::time::sleep(INACTIVITY_TIMEOUT) => {
                    if last_message_time.elapsed() >= INACTIVITY_TIMEOUT {
                        println!("Inactivity detected, flushing remaining messages...");
                        let received = processing_context.messages_received.load(Ordering::Relaxed);
                        let sent = processing_context.messages_sent.load(Ordering::Relaxed);
                        println!("Messages received: {}, Messages sent to Kafka: {}, Messages in flight: {}",
                            received, sent, received - sent);
                        futures::future::join_all(handles.drain(..)).await;
                        let final_received = processing_context.messages_received.load(Ordering::Relaxed);
                        let final_sent = processing_context.messages_sent.load(Ordering::Relaxed);
                        println!("After flush - Messages received: {}, Messages sent to Kafka: {}, Messages in flight: {}",
                            final_received, final_sent, final_received - final_sent);
                    }
                }
            }
        }
    });

    // Main ZeroMQ receiving loop
    loop {
        let msg = socket.recv().await?;
        let default_bytes = prost::bytes::Bytes::new();
        let bytes = msg.get(0).unwrap_or(&default_bytes);
        match Signals::decode(&bytes[..]) {
            Ok(signals) => {
                context.messages_received.fetch_add(1, Ordering::Relaxed);
                if let Err(e) = tx.send(signals).await {
                    eprintln!("Failed to send message to processing task: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to decode protobuf message: {}", e);
            }
        }

        // Print statistics periodically after a certain number of messages
        let received = context.messages_received.load(Ordering::Relaxed);
        if received % 10_000 == 0 {
            let sent = context.messages_sent.load(Ordering::Relaxed);
            let current_capacity = tx.capacity();
            let now = Instant::now();
            let mut last_time = context.last_time.lock().unwrap();
            let elapsed = now.duration_since(*last_time);
            let interval_count = received - context.last_count.swap(received, Ordering::Relaxed);
            let interval_speed = interval_count as f64 / elapsed.as_secs_f64();
            *last_time = now;
            drop(last_time);

            println!(
                "Messages received: {}, sent: {}, in flight: {}, capacity: {} ({:.1}%)\nInterval speed: {:.2} msg/s",
                received,
                sent,
                received - sent,
                current_capacity,
                (current_capacity as f64 / config.channel_buffer_size as f64) * 100.0,
                interval_speed
            );

            if current_capacity < LOW_CAPACITY_THRESHOLD {
                println!("Warning: Channel capacity running low - possible backpressure");
            }
        }
    }
}
