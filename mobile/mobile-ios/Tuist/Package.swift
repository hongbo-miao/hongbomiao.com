// swift-tools-version: 6.0
import PackageDescription

#if TUIST
  import struct ProjectDescription.PackageSettings

  let packageSettings = PackageSettings(
    // Customize the product types for specific package product
    // Default is .staticFramework
    // productTypes: ["Alamofire": .framework,]
    productTypes: [:]
  )
#endif

let package = Package(
  name: "mobile-ios",
  dependencies: [
    .package(url: "https://github.com/argmaxinc/WhisperKit", from: "0.14.1"),
    .package(url: "https://github.com/ml-explore/mlx-swift-examples", from: "2.25.7"),
  ]
)
