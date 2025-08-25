use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

pub async fn process_police_audio_stream(
    police_stream_id: String,
    stream_url: String,
    audio_sender: broadcast::Sender<Vec<u8>>,
) {
    info!(
        "Starting police audio stream processing for {}",
        police_stream_id
    );

    loop {
        let mut ffmpeg_child = match Command::new("ffmpeg")
            .args([
                "-i",
                &stream_url,
                "-f",
                "wav",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                error!("Failed to spawn ffmpeg for {}: {}", police_stream_id, error);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        let stdout = ffmpeg_child.stdout.take().expect("Failed to get stdout");
        let stderr = ffmpeg_child.stderr.take().expect("Failed to get stderr");

        // Handle stderr in a separate task
        let police_stream_id_clone = police_stream_id.clone();
        tokio::spawn(async move {
            let stderr_reader = BufReader::new(stderr);
            let mut lines = stderr_reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.contains("error") || line.contains("Error") {
                    warn!("FFmpeg stderr for {}: {}", police_stream_id_clone, line);
                }
            }
        });

        // Read audio data from stdout
        let mut stdout_reader = BufReader::new(stdout);
        let mut buffer = vec![0u8; 4096];

        loop {
            match stdout_reader.read(&mut buffer).await {
                Ok(0) => {
                    warn!("FFmpeg stdout closed for {}", police_stream_id);
                    break;
                }
                Ok(bytes_read) => {
                    let audio_chunk = buffer[..bytes_read].to_vec();
                    if let Err(error) = audio_sender.send(audio_chunk) {
                        if audio_sender.receiver_count() == 0 {
                            info!("No receivers for {}, stopping stream", police_stream_id);
                            break;
                        }
                        warn!(
                            "Failed to send audio chunk for {}: {}",
                            police_stream_id, error
                        );
                    }
                }
                Err(error) => {
                    error!(
                        "Error reading from FFmpeg stdout for {}: {}",
                        police_stream_id, error
                    );
                    break;
                }
            }
        }

        // Clean up the process
        if let Err(error) = ffmpeg_child.kill().await {
            warn!(
                "Failed to kill ffmpeg process for {}: {}",
                police_stream_id, error
            );
        }

        // Wait before retrying
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}
