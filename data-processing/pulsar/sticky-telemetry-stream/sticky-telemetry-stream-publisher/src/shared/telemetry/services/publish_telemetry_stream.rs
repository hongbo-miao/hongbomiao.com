use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use pulsar::producer::SendFuture;
use pulsar::{Executor, Producer};
use tracing::info;

use crate::shared::telemetry::models::telemetry_record::TelemetryRecord;

pub async fn publish_telemetry_stream<E: Executor>(
    producer: &mut Producer<E>,
    publisher_id: &str,
    publish_interval: Duration,
) -> Result<()> {
    let mut published_sample_count: u64 = 0;
    let mut sample_index: u64 = 0;
    let mut pending_send_future_list: Vec<SendFuture> = Vec::new();

    loop {
        let timestamp_ns: i64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| anyhow::anyhow!("System clock is before UNIX epoch: {error}"))?
            .as_nanos()
            .try_into()
            .map_err(|error| anyhow::anyhow!("Timestamp nanoseconds overflow i64: {error}"))?;

        let telemetry = TelemetryRecord {
            publisher_id: publisher_id.to_string(),
            timestamp_ns,
            temperature_c: Some(sample_index as f64),
            humidity_pct: Some(sample_index as f64),
        };

        let send_future = producer
            .send_non_blocking(telemetry)
            .await
            .map_err(|error| anyhow::anyhow!("Failed to send message: {error}"))?;

        pending_send_future_list.push(send_future);
        published_sample_count += 1;

        if published_sample_count.is_multiple_of(10) {
            for send_future in pending_send_future_list.drain(..) {
                send_future.await.map_err(|error| {
                    anyhow::anyhow!("Failed to confirm message delivery: {error}")
                })?;
            }
            info!("Publisher {publisher_id}: published {published_sample_count} telemetry samples");
        }

        sample_index += 1;
        tokio::time::sleep(publish_interval).await;
    }
}
