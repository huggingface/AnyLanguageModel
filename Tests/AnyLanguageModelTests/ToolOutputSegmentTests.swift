import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

@Suite("Tool output segments")
struct ToolOutputSegmentTests {
    @Test func stringOutputIsPlainText() async throws {
        let output = "Sunny\nHigh of \"21\""
        let segments = try await StringTool(output: output).makeOutputSegments(
            from: GeneratedContent("read")
        )
        guard case .text(let text) = try #require(segments.first) else {
            Issue.record("Expected a text segment")
            return
        }
        #expect(segments.count == 1)
        #expect(text.content == output)
    }

    @Test func structuredOutputIsAStructure() async throws {
        let segments = try await StructuredTool().makeOutputSegments(from: GeneratedContent("read"))
        guard case .structure(let structure) = try #require(segments.first) else {
            Issue.record("Expected a structure segment")
            return
        }
        #expect(segments.count == 1)
        #expect(structure.source == "structured")
        #expect(structure.content == GeneratedContent(properties: ["count": 3]))
    }
}

private struct StringTool: Tool {
    let name = "string"
    let description = "Return a string"
    let output: String

    var parameters: GenerationSchema {
        String.generationSchema
    }

    func call(arguments: String) async throws -> String {
        output
    }
}

private struct StructuredTool: Tool {
    let name = "structured"
    let description = "Return structured content"

    var parameters: GenerationSchema {
        String.generationSchema
    }

    func call(arguments: String) async throws -> GeneratedContent {
        GeneratedContent(properties: ["count": 3])
    }
}
