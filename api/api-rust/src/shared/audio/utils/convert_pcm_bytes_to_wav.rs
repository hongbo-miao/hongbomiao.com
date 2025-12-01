use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;

use anyhow::{Result, bail};

pub fn convert_pcm_bytes_to_wav(pcm_data: &[u8]) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    if !pcm_data.len().is_multiple_of(2) {
        bail!(
            "PCM data length must be even (16-bit samples), got {}",
            pcm_data.len()
        );
    }

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec)?;

    // Convert bytes to i16 samples and write
    for chunk in pcm_data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(cursor.into_inner())
}
