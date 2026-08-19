use anyhow::{Result, anyhow};
use arrow::array::RecordBatch;
use arrow_flight::FlightEndpoint;
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures_util::TryStreamExt;
use tonic::transport::Channel;

use crate::shared::doris::utils::extract_backend_pod_name::extract_backend_pod_name;
use crate::shared::doris::utils::rewrite_location_uri_scheme::rewrite_location_uri_scheme;

pub async fn fetch_record_batches_from_backend(
    endpoint: &FlightEndpoint,
    fe_client: &mut FlightSqlServiceClient<Channel>,
    bearer_token: &Option<String>,
) -> Result<(Vec<RecordBatch>, Option<String>)> {
    let ticket = endpoint
        .ticket
        .clone()
        .ok_or_else(|| anyhow!("FlightEndpoint has no ticket"))?;

    let Some(location) = endpoint.location.first() else {
        // An endpoint with no location means the ticket is meant to be redeemed on the original
        // (FE) connection, per the Flight spec. This is handled explicitly rather than silently
        // returning no pod name, since that would look identical to a real isolation failure.
        let stream = fe_client
            .do_get(ticket)
            .await
            .map_err(|error| anyhow!("do_get against FE failed: {error}"))?;
        let batches = stream
            .try_collect::<Vec<RecordBatch>>()
            .await
            .map_err(|error| anyhow!("Failed to collect record batches from FE: {error}"))?;
        return Ok((batches, None));
    };

    let backend_uri = rewrite_location_uri_scheme(&location.uri)?;
    let backend_pod_name = extract_backend_pod_name(&location.uri)?;

    let channel = Channel::from_shared(backend_uri.clone())
        .map_err(|error| anyhow!("Invalid backend URI {backend_uri}: {error}"))?
        .connect()
        .await
        .map_err(|error| anyhow!("Failed to connect to backend {backend_uri}: {error}"))?;
    let mut backend_client = FlightSqlServiceClient::new(channel);
    if let Some(token) = bearer_token {
        backend_client.set_token(token.clone());
    }

    let stream = backend_client
        .do_get(ticket)
        .await
        .map_err(|error| anyhow!("do_get against backend {backend_uri} failed: {error}"))?;
    let batches = stream
        .try_collect::<Vec<RecordBatch>>()
        .await
        .map_err(|error| anyhow!("Failed to collect record batches from {backend_uri}: {error}"))?;

    Ok((batches, Some(backend_pod_name)))
}
