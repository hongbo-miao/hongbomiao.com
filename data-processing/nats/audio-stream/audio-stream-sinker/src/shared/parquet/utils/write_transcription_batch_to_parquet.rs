use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{BinaryArray, Float64Array, Int64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

pub struct TranscriptionRow {
    pub stream_id: String,
    pub timestamp_ns: i64,
    pub text: String,
    pub language: String,
    pub duration_s: f64,
    pub sample_rate_hz: u32,
    pub audio_data: Vec<u8>,
    pub audio_format: String,
}

fn build_schema() -> Schema {
    Schema::new(vec![
        Field::new("stream_id", DataType::Utf8, false),
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("duration_s", DataType::Float64, false),
        Field::new("sample_rate_hz", DataType::UInt32, false),
        Field::new("audio_data", DataType::Binary, false),
        Field::new("audio_format", DataType::Utf8, false),
    ])
}

pub fn write_transcription_batch_to_parquet(
    output_directory: &Path,
    batch_timestamp_ns: i64,
    rows: &[TranscriptionRow],
) -> Result<PathBuf> {
    fs::create_dir_all(output_directory)?;

    let file_name = format!("transcriptions_{batch_timestamp_ns}.parquet");
    let file_path = output_directory.join(&file_name);

    let schema = Arc::new(build_schema());

    let stream_id_values: Vec<&str> = rows.iter().map(|row| row.stream_id.as_str()).collect();
    let timestamp_ns_values: Vec<i64> = rows.iter().map(|row| row.timestamp_ns).collect();
    let text_values: Vec<&str> = rows.iter().map(|row| row.text.as_str()).collect();
    let language_values: Vec<&str> = rows.iter().map(|row| row.language.as_str()).collect();
    let duration_s_values: Vec<f64> = rows.iter().map(|row| row.duration_s).collect();
    let sample_rate_hz_values: Vec<u32> = rows.iter().map(|row| row.sample_rate_hz).collect();
    let audio_data_values: Vec<&[u8]> = rows.iter().map(|row| row.audio_data.as_slice()).collect();
    let audio_format_values: Vec<&str> = rows.iter().map(|row| row.audio_format.as_str()).collect();

    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(stream_id_values)),
            Arc::new(Int64Array::from(timestamp_ns_values)),
            Arc::new(StringArray::from(text_values)),
            Arc::new(StringArray::from(language_values)),
            Arc::new(Float64Array::from(duration_s_values)),
            Arc::new(UInt32Array::from(sample_rate_hz_values)),
            Arc::new(BinaryArray::from(audio_data_values)),
            Arc::new(StringArray::from(audio_format_values)),
        ],
    )?;

    let writer_properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let file = fs::File::create(&file_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(writer_properties))?;
    writer.write(&record_batch)?;
    writer.close()?;

    Ok(file_path)
}
