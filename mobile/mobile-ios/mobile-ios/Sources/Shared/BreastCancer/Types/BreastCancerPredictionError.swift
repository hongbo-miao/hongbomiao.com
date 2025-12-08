import Foundation

enum BreastCancerPredictionError: LocalizedError {
  case modelNotFound
  case invalidFeatureValueCount(expectedCount: Int, actualCount: Int)
  case outputUnavailable

  var errorDescription: String? {
    switch self {
    case .modelNotFound:
      return "Breast cancer model is missing from the bundle"
    case .invalidFeatureValueCount(let expectedCount, let actualCount):
      return "Expected \(expectedCount) feature values but received \(actualCount)"
    case .outputUnavailable:
      return "Model output is unavailable"
    }
  }
}
