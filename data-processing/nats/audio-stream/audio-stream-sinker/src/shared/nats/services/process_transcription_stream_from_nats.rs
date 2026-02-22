use std::path::Path;
use std::time::{Duration, Instant};

use async_nats::jetstream;
use futures_util::StreamExt;
use tracing::{error, info};

use crate::config::AppConfig;
use crate::shared::parquet::utils::write_transcription_batch_to_parquet::{
    TranscriptionRow, write_transcription_batch_to_parquet,
};
use crate::transcription_capnp;

fn deserialize_transcription(payload: &[u8]) -> Result<TranscriptionRow, capnp::Error> {
    let reader =
        capnp::serialize::read_message_from_flat_slice(&mut &*payload, Default::default())?;
    let transcription = reader.get_root::<transcription_capnp::transcription::Reader>()?;

    Ok(TranscriptionRow {
        stream_id: transcription.get_stream_id()?.to_string()?,
        timestamp_ns: transcription.get_timestamp_ns(),
        text: transcription.get_text()?.to_string()?,
        language: transcription.get_language()?.to_string()?,
        duration_s: transcription.get_duration_s(),
        sample_rate_hz: transcription.get_sample_rate_hz(),
        audio_data: transcription.get_audio_data()?.to_vec(),
        audio_format: transcription.get_audio_format()?.to_string()?,
    })
}

fn flush_batch(
    output_directory: &Path,
    buffer: &mut Vec<TranscriptionRow>,
) -> Result<(), Box<dyn std::error::Error>> {
    if buffer.is_empty() {
        return Ok(());
    }

    let batch_timestamp_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let row_count = buffer.len();

    let file_path =
        write_transcription_batch_to_parquet(output_directory, batch_timestamp_ns, buffer)?;
    info!(
        "Wrote {row_count} transcription rows to Parquet: {}",
        file_path.display()
    );

    buffer.clear();
    Ok(())
}

pub async fn process_transcription_stream_from_nats() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::get();

    info!("Connecting to NATS at {}", config.nats_url);

    let nats_client = async_nats::connect(&config.nats_url).await?;
    let jetstream_context = jetstream::new(nats_client);

    info!(
        "Getting NATS stream: {} for subject filter: {}",
        config.nats_stream_name, config.subject_filter
    );

    let stream = jetstream_context
        .get_stream(&config.nats_stream_name)
        .await?;

    let consumer = stream
        .create_consumer(jetstream::consumer::pull::Config {
            durable_name: Some(config.queue_group.to_string()),
            filter_subject: config.subject_filter.to_string(),
            deliver_policy: jetstream::consumer::DeliverPolicy::New,
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        })
        .await?;

    info!(
        "Created consumer for stream: {} with subject filter: {}",
        config.nats_stream_name, config.subject_filter
    );

    let mut messages = consumer
        .stream()
        .max_messages_per_batch(100)
        .heartbeat(Duration::from_secs(5))
        .messages()
        .await?;

    let output_directory = Path::new(&config.parquet_output_directory);
    let mut buffer: Vec<TranscriptionRow> = Vec::with_capacity(config.parquet_batch_size);
    let mut pending_messages: Vec<jetstream::message::Message> =
        Vec::with_capacity(config.parquet_batch_size);
    let mut last_flush_time = Instant::now();

    info!(
        "Starting to sink transcriptions to Parquet at {} (batch_size={}, flush_interval={}s)",
        config.parquet_output_directory, config.parquet_batch_size, config.parquet_flush_interval_s
    );

    loop {
        match messages.next().await {
            Some(Ok(message)) => {
                let payload = message.payload.to_vec();

                match deserialize_transcription(&payload) {
                    Ok(row) => {
                        info!(
                            "Received transcription (stream_id={}, duration={:.2}s): {}",
                            row.stream_id, row.duration_s, row.text
                        );
                        buffer.push(row);
                        pending_messages.push(message);
                    }
                    Err(error) => {
                        error!("Failed to deserialize Transcription message: {error}");
                        // Ack deserialization failures immediately to avoid infinite redelivery
                        if let Err(ack_error) = message.ack().await {
                            error!("Failed to acknowledge message: {ack_error}");
                        }
                    }
                }

                let should_flush = buffer.len() >= config.parquet_batch_size
                    || (last_flush_time.elapsed().as_secs() >= config.parquet_flush_interval_s
                        && !buffer.is_empty());

                if should_flush {
                    match flush_batch(output_directory, &mut buffer) {
                        Ok(()) => {
                            for pending_message in pending_messages.drain(..) {
                                if let Err(error) = pending_message.ack().await {
                                    error!("Failed to acknowledge message: {error}");
                                }
                            }
                            last_flush_time = Instant::now();
                        }
                        Err(error) => {
                            error!("Failed to flush batch to Parquet: {error}");
                            // Don't ack - messages will be redelivered by NATS
                            buffer.clear();
                            pending_messages.clear();
                        }
                    }
                }
            }
            Some(Err(error)) => {
                error!("Error receiving message: {error}");
                break;
            }
            None => {
                info!("Stream ended");
                break;
            }
        }
    }

    match flush_batch(output_directory, &mut buffer) {
        Ok(()) => {
            for pending_message in pending_messages.drain(..) {
                if let Err(error) = pending_message.ack().await {
                    error!("Failed to acknowledge message: {error}");
                }
            }
        }
        Err(error) => {
            error!("Failed to flush final batch to Parquet: {error}");
        }
    }

    Ok(())
}
