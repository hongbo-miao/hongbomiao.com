use anyhow::Result;
use nalgebra::Matrix3;
use rerun as rr;

pub fn log_camera_calibration_to_rerun(
    recording: &rr::RecordingStream,
    entity_path: &str,
    camera_intrinsic: &Matrix3<f64>,
    rotation_wxyz: [f64; 4],
    translation_xyz: [f64; 3],
    image_width: u32,
    image_height: u32,
) -> Result<()> {
    // Convert rotation from wxyz to xyzw for Rerun and cast to f32
    let rotation_xyzw = [
        rotation_wxyz[1] as f32,
        rotation_wxyz[2] as f32,
        rotation_wxyz[3] as f32,
        rotation_wxyz[0] as f32,
    ];

    let translation_xyz_f32 =
        translation_xyz.map(|translation_component| translation_component as f32);

    // Log camera transform
    // nuScenes provides sensor-to-vehicle transform, use ParentFromChild relation
    recording.log_static(
        entity_path,
        &rr::Transform3D::from_translation_rotation(
            translation_xyz_f32,
            rr::Quaternion::from_xyzw(rotation_xyzw),
        )
        .with_relation(rr::TransformRelation::ParentFromChild),
    )?;

    // Log pinhole camera parameters
    let intrinsic_f32: [[f32; 3]; 3] = camera_intrinsic.map(|value| value as f32).into();

    recording.log_static(
        entity_path,
        &rr::Pinhole::new(intrinsic_f32).with_resolution([image_width as f32, image_height as f32]),
    )?;

    Ok(())
}
