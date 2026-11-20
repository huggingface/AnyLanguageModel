import Foundation
import Testing

@testable import AnyLanguageModel

/// Instructions go to Gemini in the request's `systemInstruction` field,
/// not in `contents` as something the user said.
@Suite("Gemini system instructions")
struct GeminiSystemInstructionsTests {
    private static let instructions = "You are a helpful assistant. Be concise."

    @Test(arguments: [false, true])
    func instructionsAreSentAsSystemInstruction(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [try Self.reply(parts: [["text": "Hi"]], streaming: streaming)])
        let session = LanguageModelSession(model: Self.model(transport), instructions: Self.instructions)

        if streaming {
            _ = try await session.streamResponse(to: "Hello").collect()
        } else {
            _ = try await session.respond(to: "Hello")
        }

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(Self.systemInstruction(of: body) == Self.instructions)
        let contents = try #require(body["contents"] as? [[String: Any]])
        #expect(contents.count == 1)
        #expect(contents.first?["role"] as? String == "user")
        #expect(Self.text(of: contents.first) == "Hello")
    }

    @Test(arguments: [false, true])
    func eachToolRoundSendsInstructionsAsSystemInstruction(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(
                parts: [["functionCall": ["name": "getWeather", "args": ["city": "Paris"]]]],
                streaming: streaming
            ),
            try Self.reply(parts: [["text": "Sunny"]], streaming: streaming),
        ])
        let session = LanguageModelSession(
            model: Self.model(transport),
            tools: [WeatherTool()],
            instructions: Self.instructions
        )

        if streaming {
            _ = try await session.streamResponse(to: "Hello").collect()
        } else {
            _ = try await session.respond(to: "Hello")
        }

        let bodies = try transport.bodies.map(Self.json)
        try #require(bodies.count == 2)
        for body in bodies {
            #expect(Self.systemInstruction(of: body) == Self.instructions)
            let contents = try #require(body["contents"] as? [[String: Any]])
            #expect(contents.first?["role"] as? String == "user")
            #expect(Self.text(of: contents.first) == "Hello")
            #expect(!contents.contains { Self.text(of: $0) == Self.instructions })
        }
    }

    @Test(arguments: [false, true])
    func structuredGenerationSendsInstructionsAsSystemInstruction(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [["text": #"{"forecast":"Sunny"}"#]], streaming: streaming)
        ])
        let session = LanguageModelSession(model: Self.model(transport), instructions: Self.instructions)

        if streaming {
            _ = try await session.streamResponse(to: "Hello", generating: Forecast.self).collect()
        } else {
            _ = try await session.respond(to: "Hello", generating: Forecast.self)
        }

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(Self.systemInstruction(of: body) == Self.instructions)
        let contents = try #require(body["contents"] as? [[String: Any]])
        #expect(contents.first?["role"] as? String == "user")
        #expect(Self.text(of: contents.first)?.hasPrefix("Hello") == true)
        #expect(!contents.contains { Self.text(of: $0) == Self.instructions })
    }

    @Test func instructionsEntriesAreJoinedWithoutImages() async throws {
        let transport = FakeTransport(replies: [try Self.reply(parts: [["text": "Hi"]], streaming: false)])
        let image = Transcript.ImageSegment(data: Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png")
        let transcript = Transcript(entries: [
            .instructions(
                .init(
                    segments: [.text(.init(content: "First.")), .image(image), .text(.init(content: "Second."))],
                    toolDefinitions: []
                )
            ),
            .instructions(.init(segments: [.text(.init(content: "Third."))], toolDefinitions: [])),
        ])
        let session = LanguageModelSession(model: Self.model(transport), transcript: transcript)

        _ = try await session.respond(to: "Hello")

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(Self.systemInstruction(of: body) == "First.\nSecond.\nThird.")
        let contents = try #require(body["contents"] as? [[String: Any]])
        #expect(contents.count == 1)
        #expect(Self.text(of: contents.first) == "Hello")
    }

    @Test func noInstructionsOmitsSystemInstruction() async throws {
        let transport = FakeTransport(replies: [try Self.reply(parts: [["text": "Hi"]], streaming: false)])
        let session = LanguageModelSession(model: Self.model(transport))

        _ = try await session.respond(to: "Hello")

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(body["systemInstruction"] == nil)
        let contents = try #require(body["contents"] as? [[String: Any]])
        #expect(contents.count == 1)
        #expect(Self.text(of: contents.first) == "Hello")
    }

    // MARK: Fixtures

    @Generable
    struct Forecast {
        var forecast: String
    }

    private static func model(_ transport: FakeTransport) -> GeminiLanguageModel {
        GeminiLanguageModel(apiKey: "test", model: "test", transport: transport)
    }

    private static func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The joined text parts of a content.
    private static func text(of content: [String: Any]?) -> String? {
        guard let parts = content?["parts"] as? [[String: Any]] else { return nil }
        return parts.compactMap { $0["text"] as? String }.joined()
    }

    /// The joined text parts of a request's system instruction.
    private static func systemInstruction(of body: [String: Any]) -> String? {
        text(of: body["systemInstruction"] as? [String: Any])
    }

    private static func reply(parts: [[String: Any]], streaming: Bool) throws -> FakeTransport.Reply {
        let object: [String: Any] = [
            "candidates": [["content": ["role": "model", "parts": parts], "finishReason": "STOP"]]
        ]
        let body = String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        return .init(chunks: [streaming ? "data: \(body)\n\n" : body])
    }
}
