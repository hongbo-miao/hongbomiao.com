use reqwest::Client;

pub async fn transcribe_audio(
    reqwest_client: &Client,
    speaches_base_url: &str,
    audio_data: &[u8],
    model: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(audio_data.to_vec())
                .file_name("audio.wav")
                .mime_str("audio/wav")?,
        )
        .text("model", model.to_string())
        .text("language", "en".to_string())
        .text("vad_filter", "false".to_string())
        .text("without_timestamps", "true".to_string())
        .text("response_format", "verbose_json".to_string())
        .text("timestamp_granularities[]", "word");

    let response = reqwest_client
        .post(format!("{speaches_base_url}/v1/audio/transcriptions"))
        .multipart(form)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!(
            "Transcription request failed with status: {}",
            response.status()
        )
        .into());
    }
    #[derive(serde::Deserialize)]
    struct ApiResponse {
        text: String,
    }
    let response_data: ApiResponse = response.json().await?;
    Ok(response_data.text.trim().to_string())
}
