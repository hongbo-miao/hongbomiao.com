mod config;
mod shared;

mod transcript_capnp {
    include!(concat!(env!("OUT_DIR"), "/transcript_capnp.rs"));
}

use crate::config::AppConfig;
use crate::shared::nats::services::subscribe_transcriptions_from_nats::subscribe_transcriptions_from_nats;
use anyhow::{Context, Result};
use sqlx::postgres::PgPoolOptions;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    tracing_subscriber::fmt()
        .with_max_level(config.log_level)
        .init();

    info!("Starting NATS to Postgres bridge");

    info!("Connecting to Postgres database");
    let pool = PgPoolOptions::new()
        .max_connections(config.postgres_max_connection_count.into())
        .connect(&config.postgres_url)
        .await
        .context("Failed to connect to Postgres database")?;
    info!("Successfully connected to Postgres database");

    match subscribe_transcriptions_from_nats(pool).await {
        Ok(()) => {
            info!("NATS to Postgres bridge completed successfully");
            Ok(())
        }
        Err(error) => {
            error!("Failed to run NATS to Postgres bridge: {error}");
            Err(error.context("NATS to Postgres bridge failed"))
        }
    }
}
