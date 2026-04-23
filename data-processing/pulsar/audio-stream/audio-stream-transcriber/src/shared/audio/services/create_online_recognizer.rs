use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};

pub fn create_online_recognizer(zipformer_model_dir: &str) -> OnlineRecognizer {
    let mut config = OnlineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(format!(
        "{zipformer_model_dir}/encoder-epoch-99-avg-1.int8.onnx"
    ));
    config.model_config.transducer.decoder =
        Some(format!("{zipformer_model_dir}/decoder-epoch-99-avg-1.onnx"));
    config.model_config.transducer.joiner = Some(format!(
        "{zipformer_model_dir}/joiner-epoch-99-avg-1.int8.onnx"
    ));
    config.model_config.tokens = Some(format!("{zipformer_model_dir}/tokens.txt"));
    config.model_config.num_threads = 1;
    config.enable_endpoint = true;
    config.decoding_method = Some("greedy_search".to_string());

    OnlineRecognizer::create(&config)
        .expect("Failed to create Zipformer online recognizer — check ZIPFORMER_MODEL_DIR")
}
