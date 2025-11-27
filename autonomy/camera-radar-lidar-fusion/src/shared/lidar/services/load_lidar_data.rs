use anyhow::{Context, Result, bail};
use nalgebra::{Matrix4xX, Vector4};
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn load_lidar_data<P: AsRef<Path>>(lidar_file_path: P) -> Result<Matrix4xX<f32>> {
    let mut file = File::open(lidar_file_path.as_ref()).with_context(|| {
        format!(
            "Failed to open lidar file {}",
            lidar_file_path.as_ref().display()
        )
    })?;

    let mut data_bytes: Vec<u8> = Vec::new();
    file.read_to_end(&mut data_bytes).with_context(|| {
        format!(
            "Failed to read lidar file {}",
            lidar_file_path.as_ref().display()
        )
    })?;

    const BYTES_PER_POINT: usize = 20;

    if data_bytes.len() % BYTES_PER_POINT != 0 {
        bail!(
            "Lidar file size {} is not a multiple of {} bytes per point",
            data_bytes.len(),
            BYTES_PER_POINT
        );
    }

    let point_count = data_bytes.len() / BYTES_PER_POINT;

    if point_count == 0 {
        return Ok(Matrix4xX::zeros(0));
    }

    let mut columns: Vec<Vector4<f32>> = Vec::with_capacity(point_count);

    for point_index in 0..point_count {
        let base = point_index * BYTES_PER_POINT;

        let x_bytes: [u8; 4] = data_bytes[base..base + 4]
            .try_into()
            .context("Failed to read x bytes")?;
        let y_bytes: [u8; 4] = data_bytes[base + 4..base + 8]
            .try_into()
            .context("Failed to read y bytes")?;
        let z_bytes: [u8; 4] = data_bytes[base + 8..base + 12]
            .try_into()
            .context("Failed to read z bytes")?;
        let intensity_bytes: [u8; 4] = data_bytes[base + 12..base + 16]
            .try_into()
            .context("Failed to read intensity bytes")?;

        let x = f32::from_le_bytes(x_bytes);
        let y = f32::from_le_bytes(y_bytes);
        let z = f32::from_le_bytes(z_bytes);
        let intensity = f32::from_le_bytes(intensity_bytes);

        columns.push(Vector4::new(x, y, z, intensity));
    }

    Ok(Matrix4xX::from_columns(&columns))
}
