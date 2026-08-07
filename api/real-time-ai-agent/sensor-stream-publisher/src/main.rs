mod fluss;
mod sensor;

use ::fluss::client::FlussConnection;
use ::fluss::config::Config;
use anyhow::Context;

use crate::fluss::services::create_sensor_tables::create_sensor_tables;
use crate::fluss::services::publish_sensor_readings::publish_sensor_readings;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bootstrap_servers =
        std::env::var("FLUSS_BOOTSTRAP_SERVERS").unwrap_or_else(|_| "127.0.0.1:9124".to_string());
    let config = Config {
        bootstrap_servers,
        ..Config::default()
    };

    let connection = FlussConnection::new(config)
        .await
        .context("failed to connect to the Fluss coordinator")?;

    create_sensor_tables(&connection)
        .await
        .context("failed to create sensor tables")?;

    publish_sensor_readings(&connection)
        .await
        .context("failed while publishing sensor readings")?;

    Ok(())
}
