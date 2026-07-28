use ort::session::Session;

const MODEL_PATH: &str = "models";
const WEIGHTS_FILE_NAME: &str = "resnet18.onnx";

pub fn load_model() -> Result<Session, String> {
    let model_path = format!("{}/{}", MODEL_PATH, WEIGHTS_FILE_NAME);
    let session = Session::builder()
        .map_err(|error| format!("Failed to create session builder: {}", error))?
        .commit_from_file(&model_path)
        .map_err(|error| format!("Failed to load model from {}: {}", model_path, error))?;

    Ok(session)
}
