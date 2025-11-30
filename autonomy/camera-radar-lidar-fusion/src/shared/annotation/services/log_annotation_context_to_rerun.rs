use anyhow::Result;
use rerun as rr;

pub fn log_annotation_context_to_rerun(
    recording: &rr::RecordingStream,
    entity_path: &str,
    annotation_context: Vec<(u16, String)>,
) -> Result<()> {
    let context: Vec<_> = annotation_context
        .iter()
        .map(|(class_id, label)| (*class_id, label.as_str()))
        .collect();

    recording.log_static(entity_path, &rr::AnnotationContext::new(context))?;

    Ok(())
}
