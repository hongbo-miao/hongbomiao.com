use anyhow::{Result, anyhow};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;

pub fn encode_pcm_i16_to_flac_bytes(
    pcm_samples: &[i16],
    sample_rate_hz: u32,
    channels_number: u32,
) -> Result<Vec<u8>> {
    if channels_number == 0 {
        return Err(anyhow!("Channels number must be greater than zero"));
    }

    // flacenc expects samples as i32 interleaved by channel
    let samples_i32: Vec<i32> = pcm_samples.iter().map(|&s| s as i32).collect();

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, verify_error)| anyhow!("Config verification failed: {:?}", verify_error))?;

    let source = flacenc::source::MemSource::from_samples(
        &samples_i32,
        channels_number as usize,
        16,
        sample_rate_hz as usize,
    );

    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|error| anyhow!("Encode failed: {error}"))?;

    let mut sink = ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|error| anyhow!("Write FLAC stream failed: {error}"))?;

    Ok(sink.as_slice().to_vec())
}
