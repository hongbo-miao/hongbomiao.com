use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use livekit::options::TrackPublishOptions;
use livekit::prelude::*;
use livekit::webrtc::video_frame::{I420Buffer, VideoFrame, VideoRotation};
use livekit::webrtc::video_source::native::NativeVideoSource;
use livekit::webrtc::video_source::{RtcVideoSource, VideoResolution};
use tracing::info;

const FRAME_WIDTH: u32 = 1_280;
const FRAME_HEIGHT: u32 = 720;
const FRAMERATE_HZ: u64 = 30;
// The bar encodes the wall-clock second it was generated at, so a viewer can decode
// the bar position and subtract it from its own clock to estimate delivery latency.
// The step count is the range of that clock in seconds, and has to comfortably exceed
// the slowest path (HLS) or the reading wraps around.
const BAR_STEP_COUNT: u32 = 20;

pub async fn publish_stepping_bar_track(room: &Arc<Room>) -> Result<TrackSid> {
    let source = NativeVideoSource::new(
        VideoResolution {
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
        },
        false,
    );

    let track =
        LocalVideoTrack::create_video_track("stepping-bar", RtcVideoSource::Native(source.clone()));

    let publication = room
        .local_participant()
        .publish_track(
            LocalTrack::Video(track),
            TrackPublishOptions {
                source: TrackSource::Camera,
                ..Default::default()
            },
        )
        .await
        .map_err(|error| anyhow!("Failed to publish video track: {error}"))?;
    let track_sid = publication.sid();

    info!("Published video track {track_sid}");

    tokio::spawn(run_stepping_bar_capture_loop(source));

    Ok(track_sid)
}

async fn run_stepping_bar_capture_loop(source: NativeVideoSource) {
    let mut ticker = tokio::time::interval(Duration::from_millis(1_000 / FRAMERATE_HZ));
    let start = tokio::time::Instant::now();

    loop {
        ticker.tick().await;

        let mut frame = VideoFrame {
            rotation: VideoRotation::VideoRotation0,
            timestamp_us: start.elapsed().as_micros() as i64,
            frame_metadata: None,
            buffer: I420Buffer::new(FRAME_WIDTH, FRAME_HEIGHT),
        };

        render_stepping_bar(&mut frame.buffer, read_wall_clock_step());

        source.capture_frame(&frame);
    }
}

fn read_wall_clock_step() -> u32 {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    (unix_seconds % u64::from(BAR_STEP_COUNT)) as u32
}

fn render_stepping_bar(buffer: &mut I420Buffer, step: u32) {
    let (stride_y, stride_u, stride_v) = buffer.strides();
    let (data_y, data_u, data_v) = buffer.data_mut();

    // A bar fills exactly one slot, so a viewer can decode the step by downsampling the
    // frame to BAR_STEP_COUNT columns and picking the brightest one.
    let bar_width = FRAME_WIDTH / BAR_STEP_COUNT;
    let bar_start = bar_width * step;
    let bar_end = (bar_start + bar_width).min(FRAME_WIDTH);

    for row in 0..FRAME_HEIGHT {
        let row_start = row as usize * stride_y as usize;
        for col in 0..FRAME_WIDTH {
            data_y[row_start + col as usize] = if (bar_start..bar_end).contains(&col) {
                235
            } else {
                16
            };
        }
    }

    for value in data_u.iter_mut() {
        *value = 128;
    }
    for value in data_v.iter_mut() {
        *value = 128;
    }
    let _ = (stride_u, stride_v);
}
