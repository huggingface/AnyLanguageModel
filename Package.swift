// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "AnyLanguageModel",
    platforms: [
        .macOS(.v14),
        .macCatalyst(.v17),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1),
    ],

    products: [
        .library(
            name: "AnyLanguageModel",
            targets: ["AnyLanguageModel"]
        )
    ],
    traits: [
        .trait(name: "CoreML"),
        .trait(name: "MLX"),
        .trait(name: "Llama"),
        .trait(name: "AsyncHTTPClient"),
        .default(enabledTraits: []),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        .package(
            url: "https://github.com/mattt/EventSource",
            from: "1.3.0",
            traits: [
                .defaults,
                .trait(name: "AsyncHTTPClient", condition: .when(traits: ["AsyncHTTPClient"])),
            ]
        ),
        .package(url: "https://github.com/mattt/JSONSchema", from: "1.3.0"),
        .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "2.7484.0")),
        .package(url: "https://github.com/mattt/PartialJSONDecoder", from: "1.0.0"),
        // mlx-swift-lm must be >= 2.25.5 for ToolSpec/tool calls and UserInput(chat:processing:tools:).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.25.5"),
        .package(url: "https://github.com/swiftlang/swift-syntax", from: "600.0.0"),
        .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.24.0"),
    ],
    targets: [
        .target(
            name: "AnyLanguageModel",
            dependencies: [
                .target(name: "AnyLanguageModelMacros"),
                .product(name: "EventSource", package: "EventSource"),
                .product(name: "JSONSchema", package: "JSONSchema"),
                .product(name: "PartialJSONDecoder", package: "PartialJSONDecoder"),
                .product(
                    name: "MLXLLM",
                    package: "mlx-swift-lm",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "MLXVLM",
                    package: "mlx-swift-lm",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "MLXLMCommon",
                    package: "mlx-swift-lm",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "Transformers",
                    package: "swift-transformers",
                    condition: .when(traits: ["CoreML"])
                ),
                .product(
                    name: "LlamaSwift",
                    package: "llama.swift",
                    condition: .when(traits: ["Llama"])
                ),
                .product(
                    name: "AsyncHTTPClient",
                    package: "async-http-client",
                    condition: .when(traits: ["AsyncHTTPClient"])
                ),
            ]
        ),
        .macro(
            name: "AnyLanguageModelMacros",
            dependencies: [
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftSyntaxBuilder", package: "swift-syntax"),
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
            ]
        ),
        .testTarget(
            name: "AnyLanguageModelTests",
            dependencies: [
                "AnyLanguageModel",
                .product(
                    name: "AsyncHTTPClient",
                    package: "async-http-client",
                    condition: .when(traits: ["AsyncHTTPClient"])
                ),
            ],
        ),
    ]
)
