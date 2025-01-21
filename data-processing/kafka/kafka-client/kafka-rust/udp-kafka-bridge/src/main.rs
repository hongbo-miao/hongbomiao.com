#![forbid(unsafe_code)]

use prost::Message;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rdkafka::ClientConfig;
use schema_registry_converter::async_impl::easy_proto_raw::EasyProtoRawEncoder;
use schema_registry_converter::async_impl::schema_registry::SrSettings;
use schema_registry_converter::schema_registry_common::SubjectNameStrategy;
use socket2::{Domain, Socket, Type};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::sleep;

pub mod iot {
    include!(concat!(env!("OUT_DIR"), "/production.iot.rs"));
}
use iot::Motor;

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
        .set("queue.buffering.max.messages", "100000") // Limit in-flight messages
        .set("queue.buffering.max.kbytes", "1048576") // 1GB max memory usage
        .set("linger.ms", "2")
        .set("compression.type", "zstd")
        .create()
        .expect("Failed to create producer")
}

async fn process_message(
    context: Arc<ProcessingContext>,
    motor: Motor,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut buf = Vec::new();
    motor.encode(&mut buf)?;
    let proto_payload = context
        .encoder
        .encode(
            &buf,
            "production.iot.Motor",
            SubjectNameStrategy::TopicNameStrategy(context.topic.clone(), false),
        )
        .await
        .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    let motor_id = motor
        .id
        .clone()
        .unwrap_or_else(|| "unknown_motor".to_string());

    context
        .producer
        .send(
            FutureRecord::to(&context.topic)
                .payload(&proto_payload)
                .key(motor_id.as_bytes()),
            Timeout::After(Duration::from_secs(1)),
        )
        .await
        .map_err(|(err, _)| Box::new(err) as Box<dyn Error + Send + Sync>)?;

    context.messages_sent.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting UDP listener and Kafka producer...");

    const CHANNEL_BUFFER_SIZE: usize = 10_000_000;
    const LOW_CAPACITY_THRESHOLD: usize = CHANNEL_BUFFER_SIZE / 10; // 10% of buffer size
    const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5); // Timeout for inactivity

    // Set up UDP socket
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, None)?;
    socket.set_reuse_address(true)?;

    // Set and get receive buffer size
    let desired_recv_buf_size = 5 * 1024 * 1024; // 5MiB
    socket.set_recv_buffer_size(desired_recv_buf_size)?;
    let actual_recv_buf_size = socket.recv_buffer_size()?;
    println!("UDP receive buffer size: {} bytes", actual_recv_buf_size);

    let addr: SocketAddr = "127.0.0.1:50537".parse()?;
    socket.bind(&addr.into())?;
    let socket = UdpSocket::from_std(socket.into())?;
    println!("Listening on 127.0.0.1:50537");

    // Set up Kafka producer and Schema Registry encoder
    let bootstrap_server = "localhost:9092";
    let producer = create_producer(bootstrap_server);
    let schema_registry_url = "https://confluent-schema-registry.internal.hongbomiao.com";
    let sr_settings = SrSettings::new(schema_registry_url.to_string());
    let encoder = EasyProtoRawEncoder::new(sr_settings);
    let topic = "production.iot.motor.proto".to_string();

    // Create processing context
    let context = Arc::new(ProcessingContext {
        producer,
        encoder,
        topic,
        messages_sent: AtomicUsize::new(0),
        messages_received: AtomicUsize::new(0),
        last_count: AtomicUsize::new(0),
        last_time: Mutex::new(Instant::now()),
    });

    // Create channel for message processing
    let (tx, mut rx) = mpsc::channel::<Motor>(CHANNEL_BUFFER_SIZE);
    println!(
        "Channel buffer size: {}, Low capacity threshold: {}",
        CHANNEL_BUFFER_SIZE, LOW_CAPACITY_THRESHOLD
    );

    // Spawn message processing task with inactivity monitoring
    let processing_context = context.clone();
    tokio::spawn(async move {
        let mut last_message_time = Instant::now();
        let mut handles = Vec::new();
        loop {
            tokio::select! {
                Some(motor) = rx.recv() => {
                    let ctx = processing_context.clone();
                    last_message_time = Instant::now();
                    handles.push(tokio::spawn(async move {
                        if let Err(e) = process_message(ctx, motor).await {
                            eprintln!("Error processing message: {}", e);
                        }
                    }));
                    handles.retain(|handle| !handle.is_finished());
                    while handles.len() >= 2000 {
                        futures::future::join_all(handles.drain(..1000)).await;
                    }
                },
                _ = sleep(INACTIVITY_TIMEOUT) => {
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

    // Main UDP receiving loop
    let mut buffer = vec![0u8; 65536];
    loop {
        let (amt, _src) = socket.recv_from(&mut buffer).await?;
        match Motor::decode(&buffer[..amt]) {
            Ok(motor) => {
                context.messages_received.fetch_add(1, Ordering::Relaxed);
                if let Err(e) = tx.send(motor).await {
                    eprintln!("Failed to send message to processing task: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to decode protobuf message: {}", e);
            }
        }

        // Print statistics every 100,000 messages
        let received = context.messages_received.load(Ordering::Relaxed);
        if received % 100_000 == 0 {
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
                "Messages received: {}, sent: {}, in flight: {}, Channel capacity: {} ({:.1}%)\nInterval speed: {:.2} msg/s",
                received,
                sent,
                received - sent,
                current_capacity,
                (current_capacity as f64 / CHANNEL_BUFFER_SIZE as f64) * 100.0,
                interval_speed
            );

            if current_capacity < LOW_CAPACITY_THRESHOLD {
                println!("Warning: Channel capacity running low - possible backpressure");
            }
        }
    }
}
