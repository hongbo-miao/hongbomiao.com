// swift-tools-version: 6.0
import PackageDescription

#if TUIST
  import struct ProjectDescription.PackageSettings

  let packageSettings = PackageSettings(
    productTypes: [
      "KokoroSwift": .staticFramework,
      "MisakiSwift": .staticFramework,
      "MLXUtilsLibrary": .staticFramework,
    ]
  )
#endif

let package = Package(
  name: "mobile-ios",
  dependencies: [
    .package(url: "https://github.com/argmaxinc/WhisperKit", from: "0.15.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.29.2"),
    .package(url: "https://github.com/mlalma/kokoro-ios", from: "1.0.9"),
  ]
)
