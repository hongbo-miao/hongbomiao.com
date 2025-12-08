import Foundation
import SwiftUI

extension ContentViewModel {
  func handlePredictBreastCancerButtonTapped() {
    guard !isPredictingBreastCancer else {
      return
    }

    isPredictingBreastCancer = true
    breastCancerPredictionText = ""

    Task {
      do {
        let probability = try predictBreastCancerProbability(
          featureValues: Config.breastCancerSampleFeatureValues
        )
        let probabilityPercentage = probability * 100
        let formattedProbability = String(format: "%.2f%%", probabilityPercentage)
        let riskDescription =
          probability >= Config.breastCancerHighRiskProbabilityThreshold ? "High risk" : "Low risk"

        await MainActor.run {
          breastCancerPredictionText =
            "\(riskDescription): \(formattedProbability) predicted malignancy probability."
          isPredictingBreastCancer = false
        }
      } catch {
        await MainActor.run {
          breastCancerPredictionText = "Prediction failed: \(error.localizedDescription)"
          isPredictingBreastCancer = false
        }
      }
    }
  }
}
