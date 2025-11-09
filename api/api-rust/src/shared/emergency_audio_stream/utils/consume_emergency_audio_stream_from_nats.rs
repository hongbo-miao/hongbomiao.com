use async_nats::jetstream;
use futures_util::StreamExt;
use tokio::sync::broadcast;
use tracing::{error, info};

use crate::shared::emergency_audio_stream::constants::emergency_streams::EMERGENCY_STREAMS;

const NATS_URL: &str = "nats://localhost:4222";
const STREAM_NAME: &str = "EMERGENCY_AUDIO_STREAMS";

pub async fn consume_emergency_audio_stream_from_nats(
    emergency_stream_id: &str,
    audio_sender: &broadcast::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let nats_client = async_nats::connect(NATS_URL).await?;
    info!("Connected to NATS at {}", NATS_URL);

    let jetstream_context = jetstream::new(nats_client);

    let emergency_stream_info = EMERGENCY_STREAMS
        .get(emergency_stream_id)
        .ok_or_else(|| format!("Unknown emergency_stream_id: {}", emergency_stream_id))?;

    let stream = jetstream_context.get_stream(STREAM_NAME).await?;

    let consumer = stream
        .create_consumer(jetstream::consumer::pull::Config {
            filter_subject: emergency_stream_info.nats_subject.to_string(),
            deliver_policy: jetstream::consumer::DeliverPolicy::New,
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        })
        .await?;

    info!(
        "Consuming from NATS subject: {}",
        emergency_stream_info.nats_subject
    );

    let mut messages = consumer
        .stream()
        .max_messages_per_batch(100)
        .heartbeat(std::time::Duration::from_secs(5))
        .messages()
        .await?;

    loop {
        match messages.next().await {
            Some(Ok(message)) => {
                let chunk = message.payload.to_vec();
                let _ = audio_sender.send(chunk);
                if let Err(error) = message.ack().await {
                    error!("Failed to acknowledge message: {}", error);
                }
            }
            Some(Err(error)) => {
                error!("Error receiving message: {}", error);
                break;
            }
            None => {
                info!("Message stream ended for {}", emergency_stream_id);
                break;
            }
        }
    }

    Ok(())
}
