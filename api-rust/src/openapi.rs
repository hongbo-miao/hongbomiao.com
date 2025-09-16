use utoipa::OpenApi;

use crate::handlers;
use crate::shared::server_sent_event::types::server_sent_event_query::ServerSentEventQuery;

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::get_root::get_root,
        handlers::get_police_audio_stream::get_police_audio_stream,
        handlers::get_sse_events::get_sse_events,
    ),
    components(
        schemas(
            ServerSentEventQuery,
            handlers::get_police_audio_stream::WebSocketParams,
        )
    ),
    tags(
        (name = "health", description = "Health check endpoints"),
        (name = "streaming", description = "Real-time streaming endpoints")
    )
)]
pub struct ApiDoc;
