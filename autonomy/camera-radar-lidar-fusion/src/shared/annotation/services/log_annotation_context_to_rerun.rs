use anyhow::Result;
use rerun as rr;

pub fn log_annotation_context_to_rerun(
    recording: &rr::RecordingStream,
    entity_path: &str,
    annotation_context: &[(u16, String)],
) -> Result<()> {
    let context = annotation_context
        .iter()
        .map(|(class_id, label)| (*class_id, label.as_str()));

    recording.log_static(entity_path, &rr::AnnotationContext::new(context))?;

    Ok(())
}
