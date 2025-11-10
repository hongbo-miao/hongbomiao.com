use wtransport::endpoint::{IncomingSession, SessionRequest};

use crate::webtransport::types::emergency_stream::FireAudioStreamIdentifier;
use crate::webtransport::utils::extract_emergency_stream_id::extract_emergency_stream_id;

pub async fn handle_incoming_session(
    incoming_session: IncomingSession,
) -> anyhow::Result<(SessionRequest, FireAudioStreamIdentifier)> {
    tracing::info!("Waiting for session request...");
    let session_request = incoming_session.await?;
    let path = session_request.path().to_string();
    let authority = session_request.authority().to_string();
    tracing::info!("New session: Authority: '{authority}', Path: '{path}'");
    let emergency_audio_stream_id = extract_emergency_stream_id(&path)?;
    Ok((session_request, emergency_audio_stream_id))
}
