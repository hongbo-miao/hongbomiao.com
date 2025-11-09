use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::shared::emergency_audio_stream::utils::emergency_audio_stream_manager::FireAudioStreamState;

pub static EMERGENCY_STREAM_STATE: Lazy<Arc<FireAudioStreamState>> =
    Lazy::new(|| Arc::new(FireAudioStreamState::new()));
