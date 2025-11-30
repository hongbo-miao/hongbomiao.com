use anyhow::Result;
use rerun as rr;

pub fn log_boxes_3d_to_rerun(
    recording: &rr::RecordingStream,
    entity_path: &str,
    centers: Vec<[f32; 3]>,
    sizes: Vec<[f32; 3]>,
    quaternions: Vec<[f32; 4]>,
    class_ids: Vec<u16>,
) -> Result<()> {
    if centers.is_empty() {
        return Ok(());
    }

    recording.log(
        entity_path,
        &rr::Boxes3D::from_centers_and_half_sizes(centers, sizes)
            .with_quaternions(quaternions)
            .with_class_ids(class_ids),
    )?;

    Ok(())
}
