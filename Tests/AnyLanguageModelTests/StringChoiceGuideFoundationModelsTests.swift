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
        /// Converted for the system model, the guides become Foundation Models' own,
        /// which encode as Foundation Models' `@Generable` encodes them.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidesConvertToFoundationModelsGuides() throws {
            let converted = FoundationModels.GenerationSchema(StringChoiceGuided.generationSchema)
            let data = try JSONEncoder().encode(converted)
            let schema = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let properties = try #require(schema["properties"] as? [String: [String: Any]])
            #expect(properties["kind"] as? NSDictionary == ["const": "fixed"])
            #expect(properties["choice"] as? NSDictionary == ["type": "string", "enum": ["a", "b"]])
        }
    }
#endif
