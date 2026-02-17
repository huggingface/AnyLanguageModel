import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Transcript")
struct TranscriptTests {
    @Generable
    struct Person {
        var name: String
    }

    @Test func entryIDRoutesToAssociatedValues() throws {
        let instructions = Transcript.Instructions(
            id: "instructions-id",
            segments: [.text(.init(id: "instructions-segment", content: "Be concise"))],
            toolDefinitions: []
        )
        let prompt = Transcript.Prompt(
            id: "prompt-id",
            segments: [.text(.init(id: "prompt-segment", content: "Hello"))]
        )
        let arguments = try GeneratedContent(json: #"{"city":"Cupertino"}"#)
        let toolCall = Transcript.ToolCall(id: "call-id", toolName: "getWeather", arguments: arguments)
        let toolCalls = Transcript.ToolCalls(id: "tool-calls-id", [toolCall])
        let toolOutput = Transcript.ToolOutput(
            id: "tool-output-id",
            toolName: "getWeather",
            segments: [.text(.init(id: "tool-output-segment", content: "Sunny"))]
        )
        let response = Transcript.Response(
            id: "response-id",
            assetIDs: [],
            segments: [.text(.init(id: "response-segment", content: "Done"))]
        )

        #expect(Transcript.Entry.instructions(instructions).id == "instructions-id")
        #expect(Transcript.Entry.prompt(prompt).id == "prompt-id")
        #expect(Transcript.Entry.toolCalls(toolCalls).id == "tool-calls-id")
        #expect(Transcript.Entry.toolOutput(toolOutput).id == "tool-output-id")
        #expect(Transcript.Entry.response(response).id == "response-id")
    }

    @Test func segmentIDRoutesToAssociatedValues() throws {
        let text = Transcript.TextSegment(id: "text-id", content: "Hello")
        let structured = Transcript.StructuredSegment(
            id: "structured-id",
            source: "source",
            content: try GeneratedContent(json: #"{"ok":true}"#)
        )
        let image = Transcript.ImageSegment(id: "image-id", url: URL(string: "https://example.com/image.png")!)

        #expect(Transcript.Segment.text(text).id == "text-id")
        #expect(Transcript.Segment.structure(structured).id == "structured-id")
        #expect(Transcript.Segment.image(image).id == "image-id")
    }

    @Test func imageSourceRoundTripsForDataAndURL() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let dataSource = Transcript.ImageSegment.Source.data(Data([0xDE, 0xAD]), mimeType: "image/png")
        let encodedDataSource = try encoder.encode(dataSource)
        let decodedDataSource = try decoder.decode(Transcript.ImageSegment.Source.self, from: encodedDataSource)
        #expect(decodedDataSource == dataSource)

        let urlSource = Transcript.ImageSegment.Source.url(URL(string: "https://example.com/a.jpg")!)
        let encodedURLSource = try encoder.encode(urlSource)
        let decodedURLSource = try decoder.decode(Transcript.ImageSegment.Source.self, from: encodedURLSource)
        #expect(decodedURLSource == urlSource)
    }

    @Test func imageSourceDecodeThrowsForUnknownKind() {
        let invalid = #"{"kind":"unknown"}"#.data(using: .utf8)!
        let decoder = JSONDecoder()

        do {
            _ = try decoder.decode(Transcript.ImageSegment.Source.self, from: invalid)
            Issue.record("Expected decoding to fail for unknown kind")
        } catch let error as DecodingError {
            if case .dataCorrupted = error {
                #expect(Bool(true))
            } else {
                Issue.record("Expected dataCorrupted, got \(error)")
            }
        } catch {
            Issue.record("Expected DecodingError, got \(error)")
        }
    }

    @Test func responseFormatNameExtractsRefTypeNameOrFallsBack() {
        let refFormat = Transcript.ResponseFormat(type: Person.self)
        #expect(refFormat.name.contains("Person"))

        let inlineSchema = GenerationSchema(type: String.self, anyOf: ["a", "b"])
        let fallbackFormat = Transcript.ResponseFormat(schema: inlineSchema)
        #expect(fallbackFormat.name == "response")
    }

    @Test func responseFormatAndToolDefinitionEquatableBehavior() {
        let firstInlineSchema = GenerationSchema(type: String.self, anyOf: ["a"])
        let secondInlineSchema = GenerationSchema(type: String.self, anyOf: ["b"])

        let firstFormat = Transcript.ResponseFormat(schema: firstInlineSchema)
        let secondFormat = Transcript.ResponseFormat(schema: secondInlineSchema)
        #expect(firstFormat == secondFormat)

        let firstToolDefinition = Transcript.ToolDefinition(
            name: "tool",
            description: "desc",
            parameters: firstInlineSchema
        )
        let secondToolDefinition = Transcript.ToolDefinition(
            name: "tool",
            description: "desc",
            parameters: secondInlineSchema
        )
        #expect(firstToolDefinition == secondToolDefinition)
    }
}
