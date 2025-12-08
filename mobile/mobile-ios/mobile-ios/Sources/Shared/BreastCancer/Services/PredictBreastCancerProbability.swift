import CoreML
import Foundation

func predictBreastCancerProbability(featureValues: [Double]) throws -> Double {
  let expectedFeatureValueCount = 30
  guard featureValues.count == expectedFeatureValueCount else {
    throw BreastCancerPredictionError.invalidFeatureValueCount(
      expectedCount: expectedFeatureValueCount,
      actualCount: featureValues.count
    )
  }

  let bundle = Bundle.main
  let modelUrl: URL
  if let compiledModelUrl = bundle.url(
    forResource: "breast_cancer_catboost",
    withExtension: "mlmodelc"
  ) {
    modelUrl = compiledModelUrl
  } else if let uncompiledModelUrl = bundle.url(
    forResource: "breast_cancer_catboost",
    withExtension: "mlmodel"
  ) {
    modelUrl = try MLModel.compileModel(at: uncompiledModelUrl)
  } else {
    throw BreastCancerPredictionError.modelNotFound
  }

  let modelConfiguration = MLModelConfiguration()
  let model = try MLModel(contentsOf: modelUrl, configuration: modelConfiguration)

  var featureDictionary: [String: MLFeatureValue] = [:]
  for (index, value) in featureValues.enumerated() {
    featureDictionary["feature_\(index)"] = MLFeatureValue(double: value)
  }

  let featureProvider = try MLDictionaryFeatureProvider(dictionary: featureDictionary)
  let predictionResult = try model.prediction(from: featureProvider)

  guard
    let probabilityMultiArray = predictionResult.featureValue(for: "prediction")?.multiArrayValue,
    probabilityMultiArray.count > 0
  else {
    throw BreastCancerPredictionError.outputUnavailable
  }

  let rawLogitValue = probabilityMultiArray[0].doubleValue
  return convertLogitToProbability(logitValue: rawLogitValue)
}
