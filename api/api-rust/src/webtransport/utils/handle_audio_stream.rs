use std::sync::Arc;

use futures_util::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use wtransport::connection::Connection;

use crate::shared::fire_audio_stream::utils::fire_audio_stream_manager::{
    FireAudioStreamManager, FireAudioStreamState,
};

use crate::webtransport::types::fire_stream::FireAudioStreamIdentifier;

pub async fn handle_audio_stream(
    connection: Connection,
    fire_audio_stream_id: FireAudioStreamIdentifier,
    fire_audio_stream_state: Arc<FireAudioStreamState>,
) -> anyhow::Result<()> {
    FireAudioStreamManager::start_stream(&fire_audio_stream_state, &fire_audio_stream_id).await;
    let client_identifier =
        FireAudioStreamManager::add_client(&fire_audio_stream_state, &fire_audio_stream_id).await;

    let audio_sender =
        FireAudioStreamManager::get_audio_sender(&fire_audio_stream_state, &fire_audio_stream_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Stream sender not found"))?;

    let mut audio_stream = BroadcastStream::new(audio_sender.subscribe());

    tracing::info!(
        "{}",
        format!("Starting to stream audio data for fire_audio_stream_id: {fire_audio_stream_id}")
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
        &fire_audio_stream_state,
        &fire_audio_stream_id,
        client_identifier,
    )
    .await;
    tracing::info!("Client disconnected");
    Ok(())
}
