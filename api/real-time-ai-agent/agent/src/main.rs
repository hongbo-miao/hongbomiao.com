mod agent;
mod fluss;

use std::sync::Arc;

use ::fluss::client::FlussConnection;
use ::fluss::config::Config;
use ::fluss::metadata::TablePath;
use anyhow::Context;
use arc_swap::ArcSwap;
use rig::prelude::*;
use rig::providers::openrouter;

use crate::agent::services::sensor_context_hook::SensorContextHook;
use crate::agent::services::serve_agent_endpoints::serve_agent_endpoints;
use crate::agent::tools::lookup_sensor_status::LookupSensorStatus;
use crate::agent::tools::scan_recent_readings::ScanRecentReadings;
use crate::fluss::services::tail_sensor_readings::tail_sensor_readings;
use crate::fluss::types::live_sensor_context::LiveSensorContext;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bootstrap_servers =
        std::env::var("FLUSS_BOOTSTRAP_SERVERS").unwrap_or_else(|_| "127.0.0.1:9124".to_string());
    let openrouter_model_id = std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string());

    let config = Config {
        bootstrap_servers,
        ..Config::default()
    };
    let connection = FlussConnection::new(config)
        .await
        .context("failed to connect to the Fluss coordinator")?;
    let connection: &'static FlussConnection = Box::leak(Box::new(connection));

    let sensor_readings_table_path = TablePath::new("fluss", "sensor_readings");
    let sensor_status_table_path = TablePath::new("fluss", "sensor_status");
    let admin = connection.get_admin()?;

    let shared_live_sensor_context = Arc::new(ArcSwap::from_pointee(LiveSensorContext::default()));
    tokio::spawn({
        let shared_live_sensor_context = shared_live_sensor_context.clone();
        let sensor_readings_table_path = sensor_readings_table_path.clone();
        async move {
            if let Err(error) = tail_sensor_readings(
                connection,
                &sensor_readings_table_path,
                shared_live_sensor_context,
            )
            .await
            {
                eprintln!("Sensor readings tail task stopped: {error}");
            }
        }
    });

    let sensor_status_table = connection.get_table(&sensor_status_table_path).await?;
    let sensor_readings_table_for_scan_tool =
        connection.get_table(&sensor_readings_table_path).await?;

    let agent = openrouter::Client::from_env()?
        .agent(openrouter_model_id.as_str())
        .preamble(
            "You answer questions about a live fleet of ground sensors. \
            Live readings are injected as context on every turn - \
            trust that context over anything you might already know. Use the \
            lookup_sensor_status tool for questions about one specific sensor, \
            and scan_recent_readings for questions about raw recent data \
            points. Always state temperatures in Celsius and pressures in \
            kilopascals, matching the units in the data.",
        )
        .max_tokens(1024)
        .additional_params(serde_json::json!({ "reasoning": { "enabled": false } }))
        .tool(LookupSensorStatus::new(sensor_status_table))
        .tool(ScanRecentReadings::new(
            sensor_readings_table_for_scan_tool,
            admin,
            sensor_readings_table_path,
        ))
        .add_hook(SensorContextHook::new(shared_live_sensor_context))
        .default_max_turns(10)
        .build();

    serve_agent_endpoints(Arc::new(agent)).await
}
