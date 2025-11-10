use crate::config::AppConfig;
use crate::shared::nats::utils::process_message::process_message;
use anyhow::{Context, Result};
use async_nats::jetstream;
use futures_util::StreamExt;
use sqlx::PgPool;
use tracing::{error, info, warn};

pub async fn subscribe_transcriptions_from_nats(pool: PgPool) -> Result<()> {
    let config = AppConfig::get();

    info!(
        "Connecting to NATS server at {} for stream {}",
        config.nats_url, config.nats_stream_name
    );

    let client = async_nats::connect(&config.nats_url)
        .await
        .context("Failed to connect to NATS server")?;

    let jetstream_context = jetstream::new(client);

    let stream = jetstream_context
        .get_stream(&config.nats_stream_name)
        .await
        .context(format!("Failed to get stream: {}", config.nats_stream_name))?;

    info!(
        "Creating consumer for stream: {} with subject filter: {}",
        config.nats_stream_name, config.subject_filter
    );

    let consumer = stream
        .create_consumer(jetstream::consumer::pull::Config {
            filter_subject: config.subject_filter.clone(),
            ..Default::default()
        })
        .await
        .context("Failed to create consumer")?;

    let mut messages = consumer
        .messages()
        .await
        .context("Failed to get message stream")?;

    info!("Successfully subscribed to NATS stream. Waiting for messages...");

    while let Some(message_result) = messages.next().await {
        match message_result {
            Ok(message) => {
                match process_message(&pool, message.payload.as_ref()).await {
                    Ok(()) => {
                        if let Err(error) = message.ack().await {
                            error!("Failed to acknowledge message: {error}");
                        }
                    }
                    Err(error) => {
                        error!("Failed to process message: {error}");
                        // Still acknowledge the message to avoid reprocessing
                        if let Err(ack_error) = message.ack().await {
                            error!("Failed to acknowledge errored message: {ack_error}");
                        }
                    }
                }
            }
            Err(error) => {
                warn!("Error receiving message from NATS: {error}");
            }
        }
    }

    info!("NATS subscription ended");
    Ok(())
}
