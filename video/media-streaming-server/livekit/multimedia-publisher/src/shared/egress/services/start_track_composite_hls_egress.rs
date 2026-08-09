use anyhow::{Result, anyhow};
use livekit::prelude::TrackSid;
use livekit_api::services::egress::{EgressClient, EgressOutput, TrackCompositeOptions, encoding};
use livekit_protocol::{
    AudioCodec, S3Upload, SegmentedFileOutput, SegmentedFileProtocol, SegmentedFileSuffix,
};
use tracing::info;

use crate::config::AppConfig;
use crate::shared::egress::utils::build_session_prefix::build_session_prefix;

const SEGMENT_DURATION_SECONDS: u32 = 2;

pub struct HlsEgressSession {
    pub egress_id: String,
    pub live_playlist_key: String,
}

pub async fn start_track_composite_hls_egress(
    config: &AppConfig,
    audio_track_sid: &TrackSid,
    video_track_sid: &TrackSid,
) -> Result<HlsEgressSession> {
    let host = config
        .livekit_url
        .replacen("ws://", "http://", 1)
        .replacen("wss://", "https://", 1);

    let client =
        EgressClient::with_api_key(&host, &config.livekit_api_key, &config.livekit_api_secret);

    let session_prefix = build_session_prefix(&config.livekit_room);
    let live_playlist_key = format!("{session_prefix}/live.m3u8");

    let segments = SegmentedFileOutput {
        protocol: SegmentedFileProtocol::HlsProtocol as i32,
        filename_prefix: format!("{session_prefix}/segment"),
        playlist_name: format!("{session_prefix}/index.m3u8"),
        live_playlist_name: live_playlist_key.clone(),
        segment_duration: SEGMENT_DURATION_SECONDS,
        // A timestamp suffix has no fixed width to outgrow, unlike the default
        // zero-padded index counter.
        filename_suffix: SegmentedFileSuffix::Timestamp as i32,
        // Session metadata belongs in a database captured from the egress_ended
        // webhook, not in a file that lives in the same bucket (and shares the same
        // retention) as the media it describes.
        disable_manifest: true,
        output: Some(livekit_protocol::segmented_file_output::Output::S3(
            S3Upload {
                access_key: config.s3_access_key.clone(),
                secret: config.s3_secret_key.clone(),
                region: config.s3_region.clone(),
                endpoint: config.s3_endpoint.clone(),
                bucket: config.hls_bucket.clone(),
                force_path_style: true,
                ..Default::default()
            },
        )),
    };

    let egress_info = client
        .start_track_composite_egress(
            &config.livekit_room,
            vec![EgressOutput::Segments(segments)],
            TrackCompositeOptions {
                // HLS is served as MPEG-TS segments, which cannot carry Opus audio; the
                // H264_720P_30 preset defaults to Opus, so override it to AAC.
                encoding: encoding::EncodingOptions {
                    audio_codec: AudioCodec::Aac,
                    ..encoding::H264_720P_30
                },
                audio_track_id: audio_track_sid.to_string(),
                video_track_id: video_track_sid.to_string(),
            },
        )
        .await
        .map_err(|error| anyhow!("Failed to start track composite egress: {error}"))?;

    info!("Started HLS egress {}", egress_info.egress_id);

    Ok(HlsEgressSession {
        egress_id: egress_info.egress_id,
        live_playlist_key,
    })
}
