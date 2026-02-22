use std::io::Cursor;

use anyhow::{Result, anyhow};

pub fn decode_flac_bytes_to_pcm_i16(flac_data: &[u8]) -> Result<(Vec<i16>, u32)> {
    let cursor = Cursor::new(flac_data);
    let mut reader = claxon::FlacReader::new(cursor)
        .map_err(|error| anyhow!("Failed to create FLAC reader: {error}"))?;

    let stream_info = reader.streaminfo();
    let sample_rate_hz = stream_info.sample_rate;
    let bits_per_sample = stream_info.bits_per_sample;

    if bits_per_sample > 16 {
        return Err(anyhow!(
            "Unsupported FLAC bit depth: {bits_per_sample}. Only up to 16-bit is supported"
        ));
    }

    let pcm_samples: Vec<i16> = reader
        .samples()
        .map(|sample| sample.map(|value| value as i16))
        .collect::<Result<Vec<i16>, _>>()
        .map_err(|error| anyhow!("Failed to decode FLAC samples: {error}"))?;

    Ok((pcm_samples, sample_rate_hz))
}
