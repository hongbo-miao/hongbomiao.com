use anyhow::{Result, anyhow};
use livekit_api::services::egress::EgressClient;
use tracing::info;

use crate::config::AppConfig;

pub async fn stop_hls_egress(config: &AppConfig, egress_id: &str) -> Result<()> {
    let host = config
        .livekit_url
        .replacen("ws://", "http://", 1)
        .replacen("wss://", "https://", 1);

    let client =
        EgressClient::with_api_key(&host, &config.livekit_api_key, &config.livekit_api_secret);

    client
        .stop_egress(egress_id)
        .await
        .map_err(|error| anyhow!("Failed to stop egress {egress_id}: {error}"))?;

    info!("Stopped HLS egress {egress_id}");

    Ok(())
}
