use anyhow::{Result, anyhow};
use arrow_flight::sql::client::FlightSqlServiceClient;
use tonic::transport::Channel;

pub async fn connect_to_flight_sql_service(uri: &str) -> Result<FlightSqlServiceClient<Channel>> {
    let channel = Channel::from_shared(uri.to_string())
        .map_err(|error| anyhow!("Invalid Flight SQL URI {uri}: {error}"))?
        .connect()
        .await
        .map_err(|error| anyhow!("Failed to connect to Flight SQL service at {uri}: {error}"))?;
    Ok(FlightSqlServiceClient::new(channel))
}
