use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;

pub fn convert_pcm_bytes_to_wav(pcm_data: &[u8]) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::new(&mut cursor, spec).expect("Failed to create WAV writer");

    // Convert bytes to i16 samples and write
    for chunk in pcm_data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        writer.write_sample(sample).expect("Failed to write sample");
    }

    writer.finalize().expect("Failed to finalize WAV");
    cursor.into_inner()
}
