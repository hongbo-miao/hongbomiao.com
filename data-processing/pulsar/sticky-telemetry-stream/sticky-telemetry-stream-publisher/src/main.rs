#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod shared;

use std::time::Duration;

use anyhow::Result;
use apache_avro::AvroSchema;
use pulsar::Pulsar;
use pulsar::proto::schema::Type as SchemaType;
use tracing::info;

use crate::shared::telemetry::models::telemetry_record::TelemetryRecord;
use crate::shared::telemetry::services::publish_telemetry_stream::publish_telemetry_stream;

const PULSAR_TOPIC: &str = "persistent://public/default/sensor-telemetry-streams";
const PUBLISH_INTERVAL: Duration = Duration::from_secs(1);

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let pulsar_service_url = std::env::var("PULSAR_SERVICE_URL")
        .unwrap_or_else(|_| "pulsar://127.0.0.1:6650".to_string());
    let publisher_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

    info!("Publisher {publisher_id}: connecting to Pulsar at {pulsar_service_url}");

    let client: Pulsar<_> = Pulsar::builder(&pulsar_service_url, pulsar::TokioExecutor)
        .build()
        .await?;

    let avro_schema_json = serde_json::to_string(&TelemetryRecord::get_schema())
        .map_err(|error| anyhow::anyhow!("Failed to serialize Avro schema to JSON: {error}"))?;

    let mut producer = client
        .producer()
        .with_topic(PULSAR_TOPIC)
        .with_name(format!("publisher-{publisher_id}"))
        .with_options(pulsar::ProducerOptions {
            schema: Some(pulsar::proto::Schema {
                r#type: SchemaType::Avro as i32,
                schema_data: avro_schema_json.into_bytes(),
                ..Default::default()
            }),
            ..Default::default()
        })
        .build()
        .await?;

    info!("Publisher {publisher_id}: connected, starting telemetry stream to {PULSAR_TOPIC}");

    if let Err(error) =
        publish_telemetry_stream(&mut producer, &publisher_id, PUBLISH_INTERVAL).await
    {
        tracing::error!("Publisher {publisher_id}: telemetry streaming failed: {error}");
        return Err(error);
    }

    Ok(())
}
