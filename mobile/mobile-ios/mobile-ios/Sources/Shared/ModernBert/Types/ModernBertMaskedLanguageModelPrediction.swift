import Foundation

struct ModernBertMaskedLanguageModelPrediction {
  let tokenIdentifier: Int
  let tokenText: String
  let probability: Double
}

struct ModernBertMaskedLanguageModelPredictionGroup {
  let maskIndex: Int
  let predictions: [ModernBertMaskedLanguageModelPrediction]
}
