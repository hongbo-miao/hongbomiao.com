use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rumqttc::{AsyncClient, QoS};
use tracing::info;

use crate::shared::telemetry::models::telemetry_record::TelemetryRecord;

pub async fn publish_telemetry_stream(
    mqtt_client: &AsyncClient,
    mqtt_topic: &str,
    publisher_id: &str,
    publish_interval: Duration,
) -> Result<()> {
    let mut published_sample_count: u64 = 0;
    let mut sample_index: u64 = 0;

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

        let json_payload = serde_json::to_vec(&telemetry)
            .map_err(|error| anyhow::anyhow!("Failed to serialize telemetry to JSON: {error}"))?;

        mqtt_client
            .publish(mqtt_topic, QoS::AtLeastOnce, false, json_payload)
            .await
            .map_err(|error| anyhow::anyhow!("Failed to publish MQTT message: {error}"))?;

        published_sample_count += 1;

        if published_sample_count.is_multiple_of(10) {
            info!(
                "Publisher {publisher_id}: published {published_sample_count} telemetry samples via MQTT"
            );
        }

        sample_index += 1;
        tokio::time::sleep(publish_interval).await;
    }
}
