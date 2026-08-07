// swift-tools-version: 6.4

import PackageDescription

let package = Package(
    name: "parakeet-ane",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            exact: "0.15.5"
        )
    ],
    targets: [
        .executableTarget(
            name: "parakeet-ane",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        )
    ]
)
