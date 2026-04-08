#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod shared;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use anyhow::Result;
use rumqttc::{AsyncClient, MqttOptions};
use tracing::info;

use crate::shared::telemetry::services::publish_telemetry_stream::publish_telemetry_stream;

const MAX_CONSECUTIVE_EVENT_LOOP_ERROR_COUNT: u32 = 10;

const MQTT_TOPIC: &str = "sensor-telemetry-raw";
const PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const MQTT_CHANNEL_CAPACITY: usize = 100_000;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mqtt_broker_host =
        std::env::var("MQTT_BROKER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let mqtt_broker_port: u16 = std::env::var("MQTT_BROKER_PORT")
        .unwrap_or_else(|_| "1883".to_string())
        .parse()
        .map_err(|error| anyhow::anyhow!("Invalid MQTT_BROKER_PORT: {error}"))?;
    let publisher_id = format!("mqtt-{}", &uuid::Uuid::new_v4().to_string()[..8]);

    info!(
        "Publisher {publisher_id}: connecting to MQTT broker at {mqtt_broker_host}:{mqtt_broker_port}"
    );

    let mut mqtt_options = MqttOptions::new(
        format!("publisher-{publisher_id}"),
        &mqtt_broker_host,
        mqtt_broker_port,
    );
    mqtt_options.set_keep_alive(Duration::from_secs(30));
    mqtt_options.set_clean_session(true);

    let (mqtt_client, mut event_loop) = AsyncClient::new(mqtt_options, MQTT_CHANNEL_CAPACITY);

    let is_connected = Arc::new(AtomicBool::new(false));
    let consecutive_error_count = Arc::new(AtomicU32::new(0));
    let is_connected_for_event_loop = Arc::clone(&is_connected);
    let consecutive_error_count_for_event_loop = Arc::clone(&consecutive_error_count);
    tokio::spawn(async move {
        loop {
            match event_loop.poll().await {
                Ok(_) => {
                    if !is_connected_for_event_loop.load(Ordering::Relaxed) {
                        is_connected_for_event_loop.store(true, Ordering::Relaxed);
                        tracing::info!("MQTT connection established");
                    }
                    consecutive_error_count_for_event_loop.store(0, Ordering::Relaxed);
                }
                Err(error) => {
                    if !is_connected_for_event_loop.load(Ordering::Relaxed) {
                        tracing::warn!("MQTT not yet connected, retrying: {error}");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                    let error_count =
                        consecutive_error_count_for_event_loop.fetch_add(1, Ordering::Relaxed) + 1;
                    tracing::error!(
                        "MQTT event loop error ({error_count}/{MAX_CONSECUTIVE_EVENT_LOOP_ERROR_COUNT}): {error}"
                    );
                    if error_count >= MAX_CONSECUTIVE_EVENT_LOOP_ERROR_COUNT {
                        tracing::error!(
                            "MQTT event loop exceeded {MAX_CONSECUTIVE_EVENT_LOOP_ERROR_COUNT} consecutive errors, exiting"
                        );
                        std::process::exit(1);
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });

    info!(
        "Publisher {publisher_id}: starting telemetry stream to {MQTT_TOPIC} (connecting to broker)"
    );

    if let Err(error) =
        publish_telemetry_stream(&mqtt_client, MQTT_TOPIC, &publisher_id, PUBLISH_INTERVAL).await
    {
        tracing::error!("Publisher {publisher_id}: telemetry streaming failed: {error}");
        return Err(error);
    }

    Ok(())
}
