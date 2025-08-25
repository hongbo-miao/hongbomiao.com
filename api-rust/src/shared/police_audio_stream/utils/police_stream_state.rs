use std::sync::Arc;

use once_cell::sync::Lazy;

use crate::shared::police_audio_stream::utils::police_audio_stream_manager::PoliceAudioStreamState;

pub static POLICE_STREAM_STATE: Lazy<Arc<PoliceAudioStreamState>> =
    Lazy::new(|| Arc::new(PoliceAudioStreamState::new()));
