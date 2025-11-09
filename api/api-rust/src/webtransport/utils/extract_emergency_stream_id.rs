use crate::shared::emergency_audio_stream::constants::emergency_streams::EMERGENCY_STREAMS;
use crate::webtransport::constants::emergency_audio_stream_paths::EMERGENCY_AUDIO_STREAM_ROUTE_PREFIX;
use crate::webtransport::types::emergency_stream::FireAudioStreamIdentifier;

pub fn extract_emergency_stream_id(path: &str) -> anyhow::Result<FireAudioStreamIdentifier> {
    let identifier = path
        .strip_prefix(EMERGENCY_AUDIO_STREAM_ROUTE_PREFIX)
        .ok_or_else(|| anyhow::anyhow!("Invalid path: {path}"))?;
    if !EMERGENCY_STREAMS.contains_key(identifier) {
        return Err(anyhow::anyhow!("Unknown emergency_stream_id: {identifier}"));
    }
    Ok(identifier.to_string())
}
