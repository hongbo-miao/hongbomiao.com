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
        .external(name: "KokoroSwift"),
        .external(name: "MLXLLM"),
        .external(name: "MLXLMCommon"),
        .external(name: "MLXUtilsLibrary"),
        .external(name: "FluidAudio"),
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
