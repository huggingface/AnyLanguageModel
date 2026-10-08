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

    @Suite("GeneratedContent JSON in Foundation Models", .enabled(if: isFoundationModelsAvailable))
    struct GeneratedContentJSONFoundationModelsTests {
        /// Foundation Models writes a space after each colon and comma.
        /// Apart from that, the property order, numbers, and escaping match.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func jsonStringMatchesFoundationModels() {
            let content = AnyLanguageModel.GeneratedContent(properties: [
                "zeta": "a/b",
                "alpha": 1,
                "mid": true,
                "beta": 2.5,
                "nested": AnyLanguageModel.GeneratedContent(properties: ["y": "why", "x": "ex"]),
            ])
            let native = FoundationModels.GeneratedContent(properties: [
                "zeta": "a/b",
                "alpha": 1,
                "mid": true,
                "beta": 2.5,
                "nested": FoundationModels.GeneratedContent(properties: ["y": "why", "x": "ex"]),
            ])

            let compacted =
                native.jsonString
                .replacingOccurrences(of: ": ", with: ":")
                .replacingOccurrences(of: ", ", with: ",")
            #expect(content.jsonString == compacted)
        }
    }
#endif
