import Foundation
import MLXLLM

enum AppConfig {
  static let audioResourceExtension: String = "wav"
  static let audioResourceName: String = "audio"
  static let breastCancerHighRiskProbabilityThreshold: Double = 0.5
  // `from sklearn.datasets import load_breast_cancer; print(load_breast_cancer().data[0].tolist())`
  static let breastCancerSampleFeatureValues: [Double] = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
    8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6,
    2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
  ]
  static let kokoroVoiceStyleName: String = "af_bella"
  // https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MLXChatExample/Services/MLXService.swift
  static let largeLanguageModelRegistryConfiguration = LLMRegistry.qwen3_1_7b_4bit
  static let llmMaxTokenCount: Int = 1024
  static let llmTemperature: Float = 0.6
  static let mlxGpuMemoryLimitByteCount: Int = 500 * 1024 * 1024  // 500 MiB
  static let modernBertSampleMaskedSentence: String = "The [MASK] of [MASK] is Paris."
  static let modernBertSequenceLengthTokenCount: Int = 128
  static let modernBertTopPredictionCount: Int = 5
  static let realTimeAudioBufferFrameCount: Int = 4096
  static let realTimeAudioProcessingIntervalNanosecondCount: UInt64 = 2 * 1_000_000_000
  static let sileroVadMaxSegmentDurationS: Double = 10.0
  static let sileroVadMinSilenceDurationS: Double = 0.35
  static let sileroVadMinSpeechDurationS: Double = 0.12
  static let sileroVadSpeechThreshold: Float = 0.35
  static let diarizerClusteringThreshold: Float = 0.6
  static let diarizerMinSpeechDurationS: Float = 0.6
  static let diarizerMinEmbeddingUpdateDurationS: Float = 1.0
  static let diarizerMinSilenceGapS: Float = 0.22
  static let diarizerMinActiveFramesCount: Float = 6.0
  static let diarizerChunkDurationS: Float = 8.0
  static let diarizerChunkOverlapS: Float = 2.0
  static let diarizerExpectedSpeakerCount: Int = 2
  // https://huggingface.co/argmaxinc/whisperkit-coreml/tree/main
  static let whisperKitModelName: String = "openai_whisper-tiny.en"
}
