use anyhow::{Context, Result, bail};
use flate2::read::ZlibDecoder;
use nalgebra::{Matrix5xX, Vector5};
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

pub fn load_radar_data<P: AsRef<Path>>(radar_file_path: P) -> Result<Matrix5xX<f32>> {
    let file = File::open(radar_file_path.as_ref()).with_context(|| {
        format!(
            "Failed to open radar file {}",
            radar_file_path.as_ref().display()
        )
    })?;
    let mut reader = BufReader::new(file);

    let mut header_lines: Vec<String> = Vec::new();
    loop {
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .context("Failed to read PCD header line")?;
        if bytes == 0 {
            bail!("Unexpected end of file while reading PCD header");
        }
        let trimmed = line.trim().to_string();
        header_lines.push(trimmed.clone());
        if trimmed.to_ascii_lowercase().starts_with("data") {
            break;
        }
    }

    let mut data_bytes: Vec<u8> = Vec::new();
    reader
        .read_to_end(&mut data_bytes)
        .context("Failed to read PCD payload")?;

    let mut fields: Vec<String> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut point_count: usize = 0;
    let mut data_format = String::new();

    for line in header_lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0].to_ascii_uppercase().as_str() {
            "FIELDS" => {
                fields = parts
                    .iter()
                    .skip(1)
                    .map(|value| value.to_string())
                    .collect();
            }
            "SIZE" => {
                sizes = parts
                    .iter()
                    .skip(1)
                    .map(|value| value.parse::<usize>().context("Invalid SIZE value"))
                    .collect::<Result<Vec<_>>>()?;
            }
            "TYPE" => {
                types = parts
                    .iter()
                    .skip(1)
                    .map(|value| value.to_string())
                    .collect();
            }
            "COUNT" => {
                counts = parts
                    .iter()
                    .skip(1)
                    .map(|value| value.parse::<usize>().context("Invalid COUNT value"))
                    .collect::<Result<Vec<_>>>()?;
            }
            "POINTS" => {
                point_count = parts
                    .get(1)
                    .context("Missing POINTS value")?
                    .parse::<usize>()
                    .context("Invalid POINTS value")?;
            }
            "DATA" => {
                data_format = parts.get(1).context("Missing DATA format")?.to_string();
            }
            _ => {}
        }
    }

    if fields.is_empty() {
        bail!("PCD header missing FIELDS definition");
    }
    if sizes.len() != fields.len() {
        bail!("SIZE vector length does not match FIELDS length");
    }
    if types.len() != fields.len() {
        bail!("TYPE vector length does not match FIELDS length");
    }
    if counts.len() != fields.len() {
        counts = vec![1; fields.len()];
    }
    if point_count == 0 {
        return Ok(Matrix5xX::zeros(0));
    }

    let mut offsets: Vec<usize> = Vec::with_capacity(fields.len());
    let mut cumulative = 0usize;
    for (size, count_value) in sizes.iter().zip(counts.iter()) {
        offsets.push(cumulative);
        cumulative += size * count_value;
    }
    let point_byte_size = cumulative;

    let payload: Vec<u8> = match data_format.as_str() {
        "binary" => data_bytes,
        "binary_compressed" => {
            if data_bytes.len() < 8 {
                bail!("Compressed PCD payload too small");
            }
            let compressed_size = u32::from_le_bytes(
                data_bytes[0..4]
                    .try_into()
                    .context("Failed to read compressed size")?,
            ) as usize;
            let uncompressed_size = u32::from_le_bytes(
                data_bytes[4..8]
                    .try_into()
                    .context("Failed to read uncompressed size")?,
            ) as usize;
            let compressed_end = 8 + compressed_size;
            if data_bytes.len() < compressed_end {
                bail!("Compressed PCD payload truncated");
            }
            let compressed_slice = &data_bytes[8..compressed_end];
            let mut decoder = ZlibDecoder::new(compressed_slice);
            let mut decompressed = vec![0u8; uncompressed_size];
            decoder
                .read_exact(&mut decompressed)
                .context("Failed to decompress PCD payload")?;
            decompressed
        }
        other => {
            bail!("Unsupported PCD DATA format: {}", other);
        }
    };

    if payload.len() < point_byte_size * point_count {
        bail!("PCD payload smaller than expected for point count");
    }

    let find_field_index = |name: &str| -> Result<usize> {
        fields
            .iter()
            .position(|f| f == name)
            .with_context(|| format!("Field '{}' not found in PCD", name))
    };

    let x_index = find_field_index("x")?;
    let y_index = find_field_index("y")?;
    let z_index = find_field_index("z")?;
    let radar_cross_section_index = find_field_index("rcs")?;
    let velocity_x_index = fields
        .iter()
        .position(|field| field == "vx_comp")
        .or_else(|| fields.iter().position(|field| field == "vx"))
        .with_context(|| "Field 'vx_comp' or 'vx' not found in PCD".to_string())?;
    let velocity_y_index = fields
        .iter()
        .position(|field| field == "vy_comp")
        .or_else(|| fields.iter().position(|field| field == "vy"))
        .with_context(|| "Field 'vy_comp' or 'vy' not found in PCD".to_string())?;

    let read_f32 = |index: usize, base: usize| -> Result<f32> {
        if sizes[index] != 4 || counts[index] != 1 || types[index] != "F" {
            bail!("Field '{}' is not a single 32-bit float", fields[index]);
        }
        let offset = base + offsets[index];
        let slice = &payload[offset..offset + 4];
        let bytes: [u8; 4] = slice.try_into().context("Failed to read f32 bytes")?;
        Ok(f32::from_le_bytes(bytes))
    };

    let mut columns: Vec<Vector5<f32>> = Vec::with_capacity(point_count);
    for point_index in 0..point_count {
        let base = point_index * point_byte_size;
        let x = read_f32(x_index, base)?;
        let y = read_f32(y_index, base)?;
        let z = read_f32(z_index, base)?;
        let radar_cross_section = read_f32(radar_cross_section_index, base)?;
        let velocity_x = read_f32(velocity_x_index, base)?;
        let velocity_y = read_f32(velocity_y_index, base)?;
        let velocity = (velocity_x * velocity_x + velocity_y * velocity_y).sqrt();
        columns.push(Vector5::new(x, y, z, velocity, radar_cross_section));
    }

    Ok(Matrix5xX::from_columns(&columns))
}
