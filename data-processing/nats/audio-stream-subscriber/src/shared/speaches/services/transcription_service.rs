pub async fn make_transcription_request(
    reqwest_client: &reqwest::Client,
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
        .text("without_timestamps", "true".to_string());

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

    let response_json: serde_json::Value = response.json().await?;
    let transcription_text = response_json
        .get("text")
        .and_then(|text| text.as_str())
        .unwrap_or("")
        .to_string();

    Ok(transcription_text)
}
