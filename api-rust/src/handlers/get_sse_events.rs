use axum::{extract::Query, http::StatusCode, response::Sse};
use futures_util::stream::{self, Stream};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::shared::server_sent_event::types::server_sent_event_query::ServerSentEventQuery;
use crate::shared::server_sent_event::utils::server_sent_event_manager::SERVER_SENT_EVENT_MANAGER;

/// Server-Sent Events endpoint for real-time data streaming
#[utoipa::path(
    get,
    path = "/sse/events",
    params(
        ("message_type" = String, Query, description = "Type of messages to subscribe to"),
        ("stream" = bool, Query, description = "Enable streaming mode")
    ),
    responses(
        (status = 200, description = "SSE stream established", content_type = "text/event-stream"),
        (status = 400, description = "Invalid query parameters", body = String)
    ),
    tag = "streaming"
)]
pub async fn get_sse_events(
    Query(params): Query<ServerSentEventQuery>,
) -> Result<
    Sse<impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>,
    (StatusCode, String),
> {
    // Validate query parameters
    if params.message_type != "police_audio_transcription" || !params.stream {
        return Err((
            StatusCode::BAD_REQUEST,
            "Use message_type=police_audio_transcription&stream=true".to_string(),
        ));
    }

    let receiver = SERVER_SENT_EVENT_MANAGER.subscribe();
    let stream = BroadcastStream::new(receiver).filter_map(|result| {
        match result {
            Ok(data) => Some(Ok(axum::response::sse::Event::default().data(data))),
            Err(_) => None, // Receiver lagged, skip this message
        }
    });

    // Create initial retry event
    let retry_event = stream::once(async {
        Ok(axum::response::sse::Event::default().retry(std::time::Duration::from_millis(3000)))
    });

    let combined_stream = retry_event.chain(stream);

    Ok(Sse::new(combined_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    ))
}
