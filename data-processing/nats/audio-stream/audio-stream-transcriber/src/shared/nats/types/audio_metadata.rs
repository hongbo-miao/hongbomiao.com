pub struct AudioMetadata<'a> {
    pub sample_rate_hz: u32,
    pub audio_data: &'a [u8],
    pub audio_format: &'a str,
}
