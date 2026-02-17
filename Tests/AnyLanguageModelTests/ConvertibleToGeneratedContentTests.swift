import Foundation
import JSONSchema
import Testing

@testable import AnyLanguageModel

@Suite("ConvertibleToGeneratedContent")
struct ConvertibleToGeneratedContentTests {
    @Test func optionalNoneMapsToNullGeneratedContent() {
        let value: GeneratedContent? = nil
        #expect(value.generatedContent.kind == .null)
    }

    @Test func optionalSomeMapsToWrappedGeneratedContent() {
        let wrapped = GeneratedContent("hello")
        let value: GeneratedContent? = wrapped

        #expect(value.generatedContent == wrapped)
    }

    @Test func arrayMapsToArrayGeneratedContent() {
        let first = GeneratedContent("a")
        let second = GeneratedContent(2)
        let array = [first, second]

        #expect(array.generatedContent.kind == .array([first, second]))
    }

    @Test func defaultInstructionsAndPromptRepresentationsUseJSONString() throws {
        let content = GeneratedContent(properties: [
            "name": "AnyLanguageModel",
            "stars": 5,
        ])
        let decoder = JSONDecoder()
        let expectedValue = try decoder.decode(JSONValue.self, from: Data(content.jsonString.utf8))
        let instructionsValue = try decoder.decode(
            JSONValue.self,
            from: Data(content.instructionsRepresentation.description.utf8)
        )
        let promptValue = try decoder.decode(
            JSONValue.self,
            from: Data(content.promptRepresentation.description.utf8)
        )

        #expect(instructionsValue == expectedValue)
        #expect(promptValue == expectedValue)
    }
}
