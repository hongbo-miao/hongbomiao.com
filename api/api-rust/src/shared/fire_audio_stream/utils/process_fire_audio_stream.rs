use tokio::sync::broadcast;
use tracing::{error, info};

use crate::shared::fire_audio_stream::utils::consume_fire_audio_stream_from_nats::consume_fire_audio_stream_from_nats;

pub async fn process_fire_audio_stream(
    fire_stream_id: String,
    audio_sender: broadcast::Sender<Vec<u8>>,
) {
    info!(
        "Starting fire audio stream processing for {}",
        fire_stream_id
    );

    loop {
        match consume_fire_audio_stream_from_nats(&fire_stream_id, &audio_sender).await {
            Ok(_) => {
                info!(
                    "Fire audio stream consumer completed for {}",
                    fire_stream_id
                );
            }
            Err(error) => {
                error!(
                    "Fire audio stream consumer error for {}: {}",
                    fire_stream_id, error
                );
            }
        }

        // Wait before retrying
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}
