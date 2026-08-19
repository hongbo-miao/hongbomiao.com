use std::env;

use anyhow::{Context, Result};

pub struct AppConfig {
    pub flight_sql_uri: String,
    pub user: String,
    pub password: String,
    pub compute_group: Option<String>,
    pub should_show_backends: bool,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        Ok(AppConfig {
            flight_sql_uri: env::var("DORIS_FLIGHT_SQL_URI")
                .unwrap_or_else(|_| "http://doris-fe-0.doris-fe-internal:8070".to_string()),
            user: env::var("DORIS_USER").context("DORIS_USER must be set")?,
            password: env::var("DORIS_PASSWORD").context("DORIS_PASSWORD must be set")?,
            compute_group: env::var("DORIS_COMPUTE_GROUP").ok(),
            should_show_backends: env::var("DORIS_SHOULD_SHOW_BACKENDS")
                .map(|value| value == "true")
                .unwrap_or(false),
        })
    }
}
