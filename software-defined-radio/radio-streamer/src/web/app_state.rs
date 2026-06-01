//! Shared state handed to every axum handler: the list of channel endpoints
//! the listener can subscribe to.

use std::sync::Arc;

use crate::channel::ChannelEndpoint;

#[derive(Clone)]
pub struct AppState {
    pub channels: Arc<Vec<ChannelEndpoint>>,
}
