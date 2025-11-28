use anyhow::Result;
use rerun as rr;

pub fn log_ego_vehicle_to_rerun(
    recording: &rr::RecordingStream,
    entity_path: &str,
    vehicle_half_length: f32,
    vehicle_half_width: f32,
    vehicle_half_height: f32,
    vehicle_elevation: f32,
) -> Result<()> {
    // Create a solid box mesh for the vehicle body
    // Box vertices (8 corners)
    let x_min = -vehicle_half_length;
    let x_max = vehicle_half_length;
    let y_min = -vehicle_half_width;
    let y_max = vehicle_half_width;
    let z_min = vehicle_elevation - vehicle_half_height;
    let z_max = vehicle_elevation + vehicle_half_height;

    #[rustfmt::skip]
    let vertices = vec![
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], // Bottom face
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max], // Top face
    ];

    // Triangle indices for box faces (12 triangles, 2 per face)
    #[rustfmt::skip]
    let indices = vec![
        // Bottom face
        [0, 1, 2], [0, 2, 3],
        // Top face
        [4, 6, 5], [4, 7, 6],
        // Front face (positive X)
        [1, 5, 6], [1, 6, 2],
        // Back face (negative X)
        [0, 3, 7], [0, 7, 4],
        // Right face (positive Y)
        [2, 6, 7], [2, 7, 3],
        // Left face (negative Y)
        [0, 4, 5], [0, 5, 1],
    ];

    // Solid blue color for all vertices
    let vertex_colors = vec![[100u8, 150u8, 255u8, 255u8]; 8];

    recording.log(
        format!("{}/body", entity_path),
        &rr::Mesh3D::new(vertices)
            .with_triangle_indices(indices)
            .with_vertex_colors(vertex_colors),
    )?;

    Ok(())
}
