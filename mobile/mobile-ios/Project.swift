import ProjectDescription

let project = Project(
  name: "mobile-ios",
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
      dependencies: []
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
