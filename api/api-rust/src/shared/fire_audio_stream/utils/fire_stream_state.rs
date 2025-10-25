use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::shared::fire_audio_stream::utils::fire_audio_stream_manager::FireAudioStreamState;

pub static FIRE_STREAM_STATE: Lazy<Arc<FireAudioStreamState>> =
    Lazy::new(|| Arc::new(FireAudioStreamState::new()));
