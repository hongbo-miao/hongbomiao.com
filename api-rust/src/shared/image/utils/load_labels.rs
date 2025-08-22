use std::fs;

const MODEL_PATH: &str = "models";
const LABELS_FILE_NAME: &str = "labels.txt";

pub fn load_labels() -> Result<Vec<String>, String> {
    let content = fs::read_to_string(format!("{}/{}", MODEL_PATH, LABELS_FILE_NAME))
        .map_err(|error| format!("Failed to read labels file: {}", error))?;
    Ok(content.lines().map(String::from).collect())
}
