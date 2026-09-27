// swift-tools-version: 5.9
import PackageDescription

// Package root is the directory that contains this file, src/, and ios/.
// Paths below are relative to that root, which is native/ in this repository
// and the repository root of the Swift package mirror.

let package = Package(
    name: "ExigeScannerCore",
    platforms: [
        .iOS(.v13),
        .macOS(.v12),
    ],
    products: [
        .library(name: "ExigeScannerCore", targets: ["ExigeScannerCore"]),
    ],
    targets: [
        .target(
            name: "CScannerCore",
            path: "src",
            sources: ["scanner_core.cpp"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("."),
            ]
        ),
        .target(
            name: "ExigeScannerCore",
            dependencies: ["CScannerCore"],
            path: "ios/Sources/ExigeScannerCore",
            resources: [
                .copy("Models"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
