use async_nats::jetstream;
use futures_util::StreamExt;
use tokio::sync::broadcast;
use tracing::{error, info};

use crate::shared::fire_audio_stream::constants::fire_streams::FIRE_STREAMS;

const NATS_URL: &str = "nats://localhost:4222";
const STREAM_NAME: &str = "FIRE_AUDIO_STREAMS";

pub async fn process_fire_audio_stream(
    fire_stream_id: String,
    audio_sender: broadcast::Sender<Vec<u8>>,
) {
    info!(
        "Starting fire audio stream processing for {}",
        fire_stream_id
    );

    loop {
        match connect_and_consume(&fire_stream_id, &audio_sender).await {
            Ok(_) => {
                info!(
                    "Fire audio stream consumer completed for {}",
                    fire_stream_id
                );
            }
            Err(error) => {
                error!(
                    "Fire audio stream consumer error for {}: {}",
                    fire_stream_id, error
                );
            }
        }

        // Wait before retrying
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}

async fn connect_and_consume(
    fire_stream_id: &str,
    audio_sender: &broadcast::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Connect to NATS
    let nats_client = async_nats::connect(NATS_URL).await?;
    info!("Connected to NATS at {}", NATS_URL);

    // Get JetStream context
    let jetstream_context = jetstream::new(nats_client);

    // Get the stream info
    let fire_stream_info = FIRE_STREAMS
        .get(fire_stream_id)
        .ok_or_else(|| format!("Unknown fire_stream_id: {}", fire_stream_id))?;

    // Get or create consumer
    let stream = jetstream_context.get_stream(STREAM_NAME).await?;

    let consumer = stream
        .create_consumer(jetstream::consumer::pull::Config {
            filter_subject: fire_stream_info.nats_subject.to_string(),
            deliver_policy: jetstream::consumer::DeliverPolicy::New,
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        })
        .await?;

    info!(
        "Consuming from NATS subject: {}",
        fire_stream_info.nats_subject
    );

    // Create message stream with continuous delivery
    // Use stream() with heartbeat to keep connection alive
    let mut messages = consumer
        .stream()
        .max_messages_per_batch(100)
        .heartbeat(std::time::Duration::from_secs(5))
        .messages()
        .await?;

    // Process messages continuously
    loop {
        match messages.next().await {
            Some(Ok(message)) => {
                let chunk = message.payload.to_vec();

                // Acknowledge the message
                if let Err(error) = message.ack().await {
                    error!("Failed to acknowledge message: {}", error);
                }

                // Broadcast audio chunk to WebSocket clients
                if audio_sender.receiver_count() > 0 {
                    let _ = audio_sender.send(chunk);
                }
            }
            Some(Err(error)) => {
                error!("Error receiving message: {}", error);
                break;
            }
            None => {
                info!("Message stream ended for {}", fire_stream_id);
                break;
            }
        }
    }

    Ok(())
}
