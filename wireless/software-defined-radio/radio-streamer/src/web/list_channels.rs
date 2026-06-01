//! `GET /channels` - returns the channel list as JSON so the listener page can
//! build its UI dynamically.

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::web::app_state::AppState;

#[derive(Serialize)]
pub struct ChannelDescription {
    pub id: usize,
    pub name: String,
    pub freq: u64,
    pub sample_rate: u32,
}

pub async fn list_channels(State(state): State<AppState>) -> Json<Vec<ChannelDescription>> {
    let descriptions = state
        .channels
        .iter()
        .map(|channel| ChannelDescription {
            id: channel.id,
            name: channel.name.clone(),
            freq: channel.freq,
            sample_rate: channel.sample_rate,
        })
        .collect();
    Json(descriptions)
}
