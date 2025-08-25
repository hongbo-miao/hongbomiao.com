use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info};

use crate::shared::server_sent_event::types::server_sent_event_message::ServerSentEventMessage;
use crate::shared::server_sent_event::utils::server_sent_event_manager::SERVER_SENT_EVENT_MANAGER;

pub async fn broadcast_transcription_result(
    stream_id: String,
    transcription_text: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime before UNIX_EPOCH when computing SSE timestamp")
        .as_secs_f64();

    let message = ServerSentEventMessage {
        message_type: "police_audio_transcription".to_string(),
        stream_id: stream_id.clone(),
        transcription: transcription_text,
        timestamp,
    };

    match SERVER_SENT_EVENT_MANAGER.publish_json(&message).await {
        Ok(_) => {
            info!(
                "Successfully published server-sent event (SSE) message for stream {}",
                stream_id
            );
            Ok(())
        }
        Err(error) => {
            error!(
                "Failed to publish server-sent event (SSE) message: {}",
                error
            );
            Err(error)
        }
    }
}
