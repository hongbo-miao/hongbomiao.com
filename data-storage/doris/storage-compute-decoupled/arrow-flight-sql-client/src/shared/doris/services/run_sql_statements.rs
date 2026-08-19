use anyhow::{Result, anyhow};
use arrow::util::pretty::pretty_format_batches;
use arrow_flight::sql::client::FlightSqlServiceClient;
use tonic::transport::Channel;
use tracing::{info, warn};

use crate::shared::doris::services::fetch_record_batches_from_backend::fetch_record_batches_from_backend;

pub async fn run_sql_statements(
    fe_client: &mut FlightSqlServiceClient<Channel>,
    bearer_token: &Option<String>,
    statements: &[String],
    should_show_backends: bool,
) -> Result<()> {
    for statement in statements {
        info!("Running statement: {statement}");
        let flight_info = fe_client
            .execute(statement.clone(), None)
            .await
            .map_err(|error| anyhow!("Failed to execute statement '{statement}': {error}"))?;

        let mut all_batches = Vec::new();
        let mut backend_pod_names = Vec::new();
        for endpoint in &flight_info.endpoint {
            let (batches, backend_pod_name) =
                fetch_record_batches_from_backend(endpoint, fe_client, bearer_token).await?;
            all_batches.extend(batches);
            if let Some(pod_name) = backend_pod_name {
                backend_pod_names.push(pod_name);
            }
        }

        if !all_batches.is_empty() {
            let formatted_result = pretty_format_batches(&all_batches)
                .map_err(|error| anyhow!("Failed to format result batches: {error}"))?;
            info!("Result:\n{formatted_result}");
        }

        if should_show_backends {
            backend_pod_names.sort();
            backend_pod_names.dedup();
            if backend_pod_names.is_empty() {
                warn!(
                    "No backend locations were returned for this statement; the endpoint \
                    provenance proof did not run for it."
                );
            } else {
                info!("Served by backend pods: {backend_pod_names:?}");
            }
        }
    }
    Ok(())
}
