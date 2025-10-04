use ort::{Environment, ExecutionProvider, Session, SessionBuilder};

const MODEL_PATH: &str = "models";
const WEIGHTS_FILE_NAME: &str = "resnet18.onnx";

pub fn load_model() -> Result<Session, String> {
    let model_path = format!("{}/{}", MODEL_PATH, WEIGHTS_FILE_NAME);
    let environment = std::sync::Arc::new(
        Environment::builder()
            .with_name("resnet18")
            .build()
            .map_err(|error| format!("Failed to create ONNX environment: {}", error))?,
    );

    let session = SessionBuilder::new(&environment)
        .map_err(|error| format!("Failed to create session builder: {}", error))?
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .map_err(|error| format!("Failed to set execution provider: {}", error))?
        .with_model_from_file(&model_path)
        .map_err(|error| format!("Failed to load model from {}: {}", model_path, error))?;

    Ok(session)
}
