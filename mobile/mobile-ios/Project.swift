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
      "SWIFT_VERSION": .string(swiftVersion)
    ]
  ),
  targets: [
    .target(
      name: "mobile-ios",
      destinations: .iOS,
      product: .app,
      bundleId: "dev.tuist.mobile-ios",
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
        .external(name: "WhisperKit")
      ]
    ),
    .target(
      name: "mobile-iosTests",
      destinations: .iOS,
      product: .unitTests,
      bundleId: "dev.tuist.mobile-iosTests",
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
