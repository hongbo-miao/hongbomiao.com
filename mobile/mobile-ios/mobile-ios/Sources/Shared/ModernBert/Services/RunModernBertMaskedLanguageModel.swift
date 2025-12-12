import CoreML
import Foundation
import Hub
import Tokenizers

func runModernBertMaskedLanguageModel() throws
  -> [ModernBertMaskedLanguageModelPredictionGroup]
{
  func loadTokenizerJsonConfig(resourceName: String, bundle: Bundle) throws -> Config {
    let tokenizerDirectoryUrl = bundle.url(
      forResource: "ModernBERTTokenizer",
      withExtension: nil
    )

    var resourceUrl = tokenizerDirectoryUrl?.appendingPathComponent("\(resourceName).json")
    if resourceUrl == nil
      || !(resourceUrl.map { FileManager.default.fileExists(atPath: $0.path) } ?? false)
    {
      resourceUrl = bundle.url(forResource: resourceName, withExtension: "json")
    }

    guard let resourceUrl else {
      throw ModernBertMaskedLanguageModelError.tokenizerFilesMissing
    }

    let data = try Data(contentsOf: resourceUrl)
    let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
    guard let dictionary = jsonObject as? [NSString: Any] else {
      throw ModernBertMaskedLanguageModelError.tokenizerFilesMissing
    }
    return Config(dictionary)
  }

  func padSequence(tokens: [Int], padTokenId: Int, targetLength: Int)
    -> (ids: [Int], attentionMask: [Int])
  {
    var paddedIds = Array(repeating: padTokenId, count: targetLength)
    var attentionMask = Array(repeating: 0, count: targetLength)
    for (index, token) in tokens.enumerated() {
      paddedIds[index] = token
      attentionMask[index] = 1
    }
    return (ids: paddedIds, attentionMask: attentionMask)
  }

  func makeFeatureProvider(
    inputIds: [Int],
    attentionMask: [Int],
    sequenceLength: Int
  ) throws -> MLFeatureProvider {
    let inputIdsArray = try MLMultiArray(
      shape: [1, NSNumber(value: sequenceLength)],
      dataType: .int32
    )
    let attentionMaskArray = try MLMultiArray(
      shape: [1, NSNumber(value: sequenceLength)],
      dataType: .int32
    )

    for (index, value) in inputIds.enumerated() {
      inputIdsArray[[0, NSNumber(value: index)]] = NSNumber(value: value)
    }

    for (index, value) in attentionMask.enumerated() {
      attentionMaskArray[[0, NSNumber(value: index)]] = NSNumber(value: value)
    }

    return try MLDictionaryFeatureProvider(
      dictionary: [
        "input_ids": MLFeatureValue(multiArray: inputIdsArray),
        "attention_mask": MLFeatureValue(multiArray: attentionMaskArray),
      ]
    )
  }

  func buildPredictions(
    tokenizer: Tokenizer,
    logits: MLMultiArray,
    maskPositions: [Int],
    topPredictionCount: Int
  ) throws -> [ModernBertMaskedLanguageModelPredictionGroup] {
    func softmax(_ values: [Double]) -> [Double] {
      guard let maxValue = values.max() else {
        return []
      }
      let expValues = values.map { exp($0 - maxValue) }
      let sumValues = expValues.reduce(0, +)
      guard sumValues > 0 else {
        return Array(repeating: 1.0 / Double(values.count), count: values.count)
      }
      return expValues.map { $0 / sumValues }
    }

    guard logits.shape.count == 3 else {
      throw ModernBertMaskedLanguageModelError.logitsUnavailable
    }

    let sequenceLength = Int(truncating: logits.shape[1])
    let vocabularySize = Int(truncating: logits.shape[2])

    let floatPointer = logits.dataPointer.bindMemory(to: Float32.self, capacity: logits.count)

    return try maskPositions.map { maskIndex -> ModernBertMaskedLanguageModelPredictionGroup in
      guard maskIndex < sequenceLength else {
        throw ModernBertMaskedLanguageModelError.logitsUnavailable
      }

      let rowStart = maskIndex * vocabularySize
      var tokenScores: [(tokenId: Int, score: Double)] = []
      tokenScores.reserveCapacity(vocabularySize)

      for tokenId in 0..<vocabularySize {
        let logit = Double(floatPointer[rowStart + tokenId])
        tokenScores.append((tokenId, logit))
      }

      let topPredictions = tokenScores.sorted { $0.score > $1.score }.prefix(topPredictionCount)
      let probabilities = softmax(topPredictions.map(\.score))

      let predictions: [ModernBertMaskedLanguageModelPrediction] = zip(
        topPredictions, probabilities
      ).map {
        let tokenText = tokenizer.decode(tokens: [$0.tokenId]).trimmingCharacters(
          in: .whitespacesAndNewlines)
        return ModernBertMaskedLanguageModelPrediction(
          tokenIdentifier: $0.tokenId,
          tokenText: tokenText.isEmpty ? "[UNK]" : tokenText,
          probability: $1
        )
      }

      return ModernBertMaskedLanguageModelPredictionGroup(
        maskIndex: maskIndex,
        predictions: predictions
      )
    }
  }

  let bundle = Bundle.main
  let potentialModelUrls: [URL?] = [
    bundle.url(forResource: "ModernBERTMaskedLM", withExtension: "mlmodelc")
  ]
  guard let modelSourceUrl = potentialModelUrls.compactMap({ $0 }).first else {
    throw ModernBertMaskedLanguageModelError.modelNotFound
  }

  let modelUrl: URL
  if modelSourceUrl.pathExtension == "mlmodel" {
    modelUrl = try MLModel.compileModel(at: modelSourceUrl)
  } else {
    modelUrl = modelSourceUrl
  }

  let tokenizerConfig = try loadTokenizerJsonConfig(
    resourceName: "tokenizer_config",
    bundle: bundle
  )
  let tokenizerData = try loadTokenizerJsonConfig(resourceName: "tokenizer", bundle: bundle)
  let tokenizer = try AutoTokenizer.from(
    tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)

  guard let maskTokenId = tokenizer.convertTokenToId("[MASK]") else {
    throw ModernBertMaskedLanguageModelError.maskTokenIdentifierMissing
  }
  let padTokenId = tokenizer.convertTokenToId("[PAD]") ?? 0

  let sentence = AppConfig.modernBertSampleMaskedSentence
  let encodedInputIds = tokenizer.encode(text: sentence)
  let sequenceLength = AppConfig.modernBertSequenceLengthTokenCount
  let truncatedInputIds = Array(encodedInputIds.prefix(sequenceLength))
  let maskPositions = truncatedInputIds.enumerated().compactMap {
    $0.element == maskTokenId ? $0.offset : nil
  }
  guard !maskPositions.isEmpty else {
    throw ModernBertMaskedLanguageModelError.maskTokenNotPresent
  }

  let paddedInputs = padSequence(
    tokens: truncatedInputIds,
    padTokenId: padTokenId,
    targetLength: sequenceLength
  )

  let modelConfiguration = MLModelConfiguration()
  let model = try MLModel(contentsOf: modelUrl, configuration: modelConfiguration)
  let predictionResult = try model.prediction(
    from: try makeFeatureProvider(
      inputIds: paddedInputs.ids,
      attentionMask: paddedInputs.attentionMask,
      sequenceLength: sequenceLength
    )
  )

  guard
    let logitsMultiArray = predictionResult.featureValue(for: "logits")?.multiArrayValue
  else {
    throw ModernBertMaskedLanguageModelError.logitsUnavailable
  }

  return try buildPredictions(
    tokenizer: tokenizer,
    logits: logitsMultiArray,
    maskPositions: maskPositions,
    topPredictionCount: AppConfig.modernBertTopPredictionCount
  )
}

extension NSNumber {
  fileprivate var intValue: Int {
    Int(truncating: self)
  }
}
