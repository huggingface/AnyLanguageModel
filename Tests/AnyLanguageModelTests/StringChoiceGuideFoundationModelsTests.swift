import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationModels)
    import FoundationModels

    private let isFoundationModelsAvailable: Bool = {
        if #available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *) {
            return true
        }
        return false
    }()

    @Suite("String choice guides in Foundation Models", .enabled(if: isFoundationModelsAvailable))
    struct StringChoiceGuideFoundationModelsTests {
        /// The encoded properties of a schema, with their keys sorted.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        private func properties(_ schema: FoundationModels.GenerationSchema) throws -> [String: NSDictionary] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            let object = try JSONSerialization.jsonObject(with: encoder.encode(schema)) as? [String: Any]
            return try #require(object?["properties"] as? [String: NSDictionary])
        }

        /// Converted for the system model, the guides encode as Foundation Models' own guides do.
        /// That encoding differs between OS versions (macOS 26 encodes a constant as a one-choice
        /// `enum`, macOS 27 as `const`), so the expected values come from Foundation Models itself.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidesConvertToFoundationModelsGuides() throws {
            let converted = try properties(FoundationModels.GenerationSchema(StringChoiceGuided.generationSchema))

            let native = try properties(
                FoundationModels.GenerationSchema(
                    root: DynamicGenerationSchema(
                        name: "Native",
                        properties: [
                            .init(name: "kind", schema: .init(type: String.self, guides: [.constant("fixed")])),
                            .init(name: "choice", schema: .init(type: String.self, guides: [.anyOf(["a", "b"])])),
                        ]
                    ),
                    dependencies: []
                )
            )
            #expect(converted["kind"] == native["kind"])
            #expect(converted["choice"] == native["choice"])

            let dynamic = try properties(
                FoundationModels.GenerationSchema(
                    AnyLanguageModel.GenerationSchema(
                        root: AnyLanguageModel.DynamicGenerationSchema(
                            name: "Dynamic",
                            properties: [
                                .init(name: "kind", schema: .init(type: String.self, guides: [.constant("fixed")])),
                                .init(name: "choice", schema: .init(type: String.self, guides: [.anyOf(["a", "b"])])),
                            ]
                        ),
                        dependencies: []
                    )
                )
            )
            #expect(dynamic["kind"] == native["kind"])
            #expect(dynamic["choice"] == native["choice"])
        }
    }
#endif
