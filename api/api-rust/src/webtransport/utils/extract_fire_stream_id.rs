use crate::shared::fire_audio_stream::constants::fire_streams::FIRE_STREAMS;
use crate::webtransport::constants::fire_audio_stream_paths::FIRE_AUDIO_STREAM_ROUTE_PREFIX;
use crate::webtransport::types::fire_stream::FireAudioStreamIdentifier;

pub fn extract_fire_stream_id(path: &str) -> anyhow::Result<FireAudioStreamIdentifier> {
    let identifier = path
        .strip_prefix(FIRE_AUDIO_STREAM_ROUTE_PREFIX)
        .ok_or_else(|| anyhow::anyhow!("Invalid path: {path}"))?;
    if !FIRE_STREAMS.contains_key(identifier) {
        return Err(anyhow::anyhow!("Unknown fire_stream_id: {identifier}"));
    }
    Ok(identifier.to_string())
}
