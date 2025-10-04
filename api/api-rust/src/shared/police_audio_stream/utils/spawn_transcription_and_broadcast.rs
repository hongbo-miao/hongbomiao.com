use tokio::sync::broadcast;
use tracing::info;

use crate::config::AppConfig;
use crate::shared::server_sent_event::utils::broadcast_transcription_result::broadcast_transcription_result;
use crate::shared::speaches::services::transcribe_audio::transcribe_audio;

pub fn spawn_transcription_and_broadcast(
    reqwest_client: reqwest::Client,
    config: &AppConfig,
    server_sent_event_sender: broadcast::Sender<String>,
    stream_id: String,
    audio_data: Vec<u8>,
) {
    let config_clone = config.clone();
    let stream_id_clone = stream_id.clone();

    tokio::spawn(async move {
        match transcribe_audio(
            &reqwest_client,
            &config_clone.speaches_base_url,
            &audio_data,
            &config_clone.transcription_model,
        )
        .await
        {
            Ok(text) => {
                if !text.is_empty() {
                    info!(stream_id = %stream_id_clone, %text, "Transcription received");

                    // Broadcast via SSE
                    if let Err(error) =
                        broadcast_transcription_result(stream_id_clone.clone(), text.clone()).await
                    {
                        tracing::error!(%error, stream_id = %stream_id_clone, "Failed to broadcast transcription via SSE");
                    }

                    // Also send via direct broadcast for WebSocket compatibility
                    let message = format!(
                        "{{\"stream_id\":\"{}\",\"transcription\":\"{}\"}}",
                        stream_id_clone,
                        text.replace('"', "\\\"")
                    );
                    let _ = server_sent_event_sender.send(message);
                } else {
                    tracing::warn!(stream_id = %stream_id_clone, "Transcription empty string returned");
                }
            }
            Err(error) => {
                tracing::error!(%error, stream_id = %stream_id_clone, "Transcription failed or service unavailable");
            }
        }
    });
}
