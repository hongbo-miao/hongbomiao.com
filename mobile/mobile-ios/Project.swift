import Foundation
import ProjectDescription

let swiftVersion = try! String(
  contentsOfFile: ".swift-version",
  encoding: .utf8
).trimmingCharacters(in: .whitespacesAndNewlines)

let project = Project(
  name: "mobile-ios",
  settings: .settings(
    base: [
      "SWIFT_VERSION": .string(swiftVersion),
      "DEVELOPMENT_TEAM": .string("KJUGT5LXHP"),
      "ASSETCATALOG_COMPILER_GENERATE_SWIFT_ASSET_SYMBOL_EXTENSIONS": .string("YES"),
      "ENABLE_USER_SCRIPT_SANDBOXING": .string("YES"),
      "STRING_CATALOG_GENERATE_SYMBOLS": .string("YES"),
    ]
  ),
  targets: [
    .target(
      name: "mobile-ios",
      destinations: .iOS,
      product: .app,
      bundleId: "com.hongbomiao.mobile-ios",
      infoPlist: .extendingDefault(
        with: [
          "UILaunchScreen": [
            "UIColorName": "",
            "UIImageName": "",
          ]
        ]
      ),
      buildableFolders: [
        "mobile-ios/Sources",
        "mobile-ios/Resources",
      ],
      dependencies: [
        .external(name: "FluidAudio"),
        .external(name: "Hub"),
        .external(name: "KokoroSwift"),
        .external(name: "MLXLLM"),
        .external(name: "MLXLMCommon"),
        .external(name: "MLXUtilsLibrary"),
        .external(name: "StableDiffusion"),
        .external(name: "Tokenizers"),
        .external(name: "WhisperKit"),
      ]
    ),
    .target(
      name: "mobile-iosTests",
      destinations: .iOS,
      product: .unitTests,
      bundleId: "com.hongbomiao.mobile-iosTests",
      infoPlist: .default,
      buildableFolders: [
        "mobile-ios/Tests"
      ],
      dependencies: [.target(name: "mobile-ios")]
    ),
  ],
  schemes: [
    .scheme(
      name: "mobile-ios",
      shared: true,
      buildAction: .buildAction(
        targets: [
          .target("mobile-ios")
        ]
      ),
      testAction: .targets(
        [
          .testableTarget(target: "mobile-iosTests")
        ],
        options: .options(
          coverage: true,
          codeCoverageTargets: [
            .target("mobile-ios")
          ]
        )
      )
    )
  ]
)
