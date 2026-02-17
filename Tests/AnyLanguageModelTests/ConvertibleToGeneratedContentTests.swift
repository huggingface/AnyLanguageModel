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

    @Test func defaultInstructionsAndPromptRepresentationsUseJSONString() {
        let content = GeneratedContent(properties: [
            "name": "AnyLanguageModel",
            "stars": 5,
        ])

        #expect(content.instructionsRepresentation.description == content.jsonString)
        #expect(content.promptRepresentation.description == content.jsonString)
    }
}
