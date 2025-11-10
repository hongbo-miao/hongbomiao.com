use std::sync::Arc;

use futures_util::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use wtransport::connection::Connection;

use crate::shared::emergency_audio_stream::utils::emergency_audio_stream_manager::{
    FireAudioStreamManager, FireAudioStreamState,
};

use crate::webtransport::types::emergency_stream::FireAudioStreamIdentifier;

pub async fn handle_audio_stream(
    connection: Connection,
    emergency_audio_stream_id: FireAudioStreamIdentifier,
    emergency_audio_stream_state: Arc<FireAudioStreamState>,
) -> anyhow::Result<()> {
    FireAudioStreamManager::start_stream(&emergency_audio_stream_state, &emergency_audio_stream_id)
        .await;
    let client_identifier = FireAudioStreamManager::add_client(
        &emergency_audio_stream_state,
        &emergency_audio_stream_id,
    )
    .await;

    let audio_sender = FireAudioStreamManager::get_audio_sender(
        &emergency_audio_stream_state,
        &emergency_audio_stream_id,
    )
    .await
    .ok_or_else(|| anyhow::anyhow!("Stream sender not found"))?;

    let mut audio_stream = BroadcastStream::new(audio_sender.subscribe());

    tracing::info!(
        "{}",
        format!(
            "Starting to stream audio data for emergency_audio_stream_id: {emergency_audio_stream_id}"
        )
    );

    let mut stream = connection.open_uni().await?.await?;
    while let Some(result) = audio_stream.next().await {
        match result {
            Ok(audio_chunk) => {
                if let Err(_error) = stream.write_all(&audio_chunk).await {
                    tracing::info!("Client disconnected, closing stream.");
                    break;
                }
            }
            Err(error) => {
                tracing::error!("Broadcast stream error: {error}");
                break;
            }
        }
    }

    FireAudioStreamManager::remove_client(
        &emergency_audio_stream_state,
        &emergency_audio_stream_id,
        client_identifier,
    )
    .await;
    tracing::info!("Client disconnected");
    Ok(())
}
