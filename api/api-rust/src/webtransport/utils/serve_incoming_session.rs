use wtransport::endpoint::IncomingSession;

use crate::shared::emergency_audio_stream::utils::emergency_stream_state::EMERGENCY_STREAM_STATE;
use crate::webtransport::utils::handle_audio_stream::handle_audio_stream;
use crate::webtransport::utils::handle_incoming_session::handle_incoming_session;

pub async fn serve_incoming_session(incoming_session: IncomingSession) {
    let result = async {
        let (session_request, emergency_audio_stream_id) =
            handle_incoming_session(incoming_session).await?;
        let connection = session_request.accept().await?;
        tracing::info!(
            "Connection established for emergency_audio_stream_id: {emergency_audio_stream_id}"
        );
        let emergency_audio_stream_state = EMERGENCY_STREAM_STATE.clone();
        handle_audio_stream(
            connection,
            emergency_audio_stream_id,
            emergency_audio_stream_state,
        )
        .await
    }
    .await;
    if let Err(error) = result {
        tracing::error!("WebTransport session error: {error}");
    }
}
