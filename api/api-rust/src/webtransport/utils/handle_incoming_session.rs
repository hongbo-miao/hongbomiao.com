use wtransport::endpoint::{IncomingSession, SessionRequest};

use crate::webtransport::types::fire_stream::FireAudioStreamIdentifier;
use crate::webtransport::utils::extract_fire_stream_id::extract_fire_stream_id;

pub async fn handle_incoming_session(
    incoming_session: IncomingSession,
) -> anyhow::Result<(SessionRequest, FireAudioStreamIdentifier)> {
    tracing::info!("Waiting for session request...");
    let session_request = incoming_session.await?;
    let path = session_request.path().to_string();
    let authority = session_request.authority().to_string();
    tracing::info!("New session: Authority: '{authority}', Path: '{path}'");
    let fire_audio_stream_id = extract_fire_stream_id(&path)?;
    Ok((session_request, fire_audio_stream_id))
}
