#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

mod config;
mod shared;

use std::io::Read;

use anyhow::{Result, anyhow};
use tracing::info;

use crate::config::AppConfig;
use crate::shared::doris::services::connect_to_flight_sql_service::connect_to_flight_sql_service;
use crate::shared::doris::services::run_sql_statements::run_sql_statements;
use crate::shared::doris::utils::split_sql_statements::split_sql_statements;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let config = AppConfig::load()?;

    let mut sql_text = String::new();
    std::io::stdin()
        .read_to_string(&mut sql_text)
        .map_err(|error| anyhow!("Failed to read SQL from stdin: {error}"))?;

    let mut statements = split_sql_statements(&sql_text);
    if let Some(compute_group) = &config.compute_group {
        statements.insert(0, format!("use paimon_catalog@{compute_group}"));
    }

    let mut fe_client = connect_to_flight_sql_service(&config.flight_sql_uri).await?;
    info!("Connected to {}", config.flight_sql_uri);

    fe_client
        .handshake(&config.user, &config.password)
        .await
        .map_err(|error| anyhow!("Handshake with Doris FE failed: {error}"))?;
    let bearer_token = fe_client.token().cloned();

    run_sql_statements(
        &mut fe_client,
        &bearer_token,
        &statements,
        config.should_show_backends,
    )
    .await
}
