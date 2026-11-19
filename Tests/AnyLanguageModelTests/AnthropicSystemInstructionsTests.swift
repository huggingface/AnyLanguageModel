import Foundation
import Testing

@testable import AnyLanguageModel

/// Instructions go to Anthropic in the top-level `system` field,
/// because the Messages API has no `system` role.
@Suite("Anthropic system instructions")
struct AnthropicSystemInstructionsTests {
    private static let instructions = "You are a helpful assistant. Be concise."

    @Test(arguments: [false, true])
    func instructionsAreSentAsSystem(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [try Self.reply(text: "Hi", streaming: streaming)])
        let session = LanguageModelSession(model: Self.model(transport), instructions: Self.instructions)

        try await Self.respond(in: session, streaming: streaming)

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(body["system"] as? String == Self.instructions)
        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.count == 1)
        #expect(messages.first?["role"] as? String == "user")
        #expect(Self.text(of: messages.first) == "Hello")
    }

    @Test(arguments: [false, true])
    func eachToolRoundSendsInstructionsAsSystem(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(toolCallStreaming: streaming),
            try Self.reply(text: "Sunny", streaming: streaming),
        ])
        let session = LanguageModelSession(
            model: Self.model(transport),
            tools: [WeatherTool()],
            instructions: Self.instructions
        )

        try await Self.respond(in: session, streaming: streaming)

        let bodies = try transport.bodies.map(Self.json)
        try #require(bodies.count == 2)
        for body in bodies {
            #expect(body["system"] as? String == Self.instructions)
            let messages = try #require(body["messages"] as? [[String: Any]])
            #expect(messages.first?["role"] as? String == "user")
            #expect(Self.text(of: messages.first) == "Hello")
            #expect(!messages.contains { Self.text(of: $0) == Self.instructions })
        }
    }

    @Test func instructionsEntriesAreJoinedWithoutImages() async throws {
        let transport = FakeTransport(replies: [try Self.reply(text: "Hi", streaming: false)])
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

        try await Self.respond(in: session, streaming: false)

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(body["system"] as? String == "First.\nSecond.\nThird.")
        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.count == 1)
        #expect(Self.text(of: messages.first) == "Hello")
    }

    @Test func noInstructionsOmitsSystem() async throws {
        let transport = FakeTransport(replies: [try Self.reply(text: "Hi", streaming: false)])
        let session = LanguageModelSession(model: Self.model(transport))

        try await Self.respond(in: session, streaming: false)

        let body = try #require(try transport.bodies.map(Self.json).first)
        #expect(body["system"] == nil)
    }

    // MARK: Fixtures

    private static func model(_ transport: FakeTransport) -> AnthropicLanguageModel {
        AnthropicLanguageModel(apiKey: "test", model: "test", transport: transport)
    }

    private static func respond(in session: LanguageModelSession, streaming: Bool) async throws {
        if streaming {
            _ = try await session.streamResponse(to: "Hello").collect()
        } else {
            _ = try await session.respond(to: "Hello")
        }
    }

    private static func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The joined text blocks of a message.
    private static func text(of message: [String: Any]?) -> String? {
        guard let content = message?["content"] as? [[String: Any]] else { return nil }
        return content.compactMap { $0["type"] as? String == "text" ? $0["text"] as? String : nil }.joined()
    }

    private static func encode(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }

    private static func message(_ content: [[String: Any]], stopReason: String) -> [String: Any] {
        [
            "id": "m", "type": "message", "role": "assistant", "model": "test", "content": content,
            "stop_reason": stopReason, "usage": ["input_tokens": 1, "output_tokens": 1],
        ]
    }

    private static func reply(text: String, streaming: Bool) throws -> FakeTransport.Reply {
        guard streaming else {
            return .init(chunks: [try encode(message([["type": "text", "text": text]], stopReason: "end_turn"))])
        }
        let events: [[String: Any]] = [
            ["type": "message_start", "message": message([], stopReason: "end_turn")],
            ["type": "content_block_start", "index": 0, "content_block": ["type": "text", "text": ""]],
            ["type": "content_block_delta", "index": 0, "delta": ["type": "text_delta", "text": text]],
            ["type": "content_block_stop", "index": 0],
            ["type": "message_stop"],
        ]
        return .init(chunks: try events.map { "data: \(try encode($0))\n\n" })
    }

    private static func reply(toolCallStreaming streaming: Bool) throws -> FakeTransport.Reply {
        let call: [String: Any] = [
            "type": "tool_use", "id": "call_1", "name": "getWeather", "input": ["city": "Paris"],
        ]
        guard streaming else { return .init(chunks: [try encode(message([call], stopReason: "tool_use"))]) }
        let events: [[String: Any]] = [
            ["type": "message_start", "message": message([], stopReason: "tool_use")],
            [
                "type": "content_block_start", "index": 0,
                "content_block": ["type": "tool_use", "id": "call_1", "name": "getWeather", "input": [:]],
            ],
            [
                "type": "content_block_delta", "index": 0,
                "delta": ["type": "input_json_delta", "partial_json": #"{"city":"Paris"}"#],
            ],
            ["type": "content_block_stop", "index": 0],
            ["type": "message_stop"],
        ]
        return .init(chunks: try events.map { "data: \(try encode($0))\n\n" })
    }
}
