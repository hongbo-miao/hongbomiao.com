import Foundation

enum StableDiffusionGenerationError: Error {
  case resourceDirectoryMissing(directoryName: String)
  case imageNotGenerated
  case unsupportedOsVersion
}
