import Foundation

func convertLogitToProbability(logitValue: Double) -> Double {
  1.0 / (1.0 + exp(-logitValue))
}
