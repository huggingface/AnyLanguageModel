import Foundation
import Testing

@testable import AnyLanguageModel

/// What the Gemini builder sends of an earlier conversation,
/// when a response has stored text parts and thought signatures.
@Suite("Gemini assistant history")
struct GeminiAssistantHistoryTests {
    private static func model(_ transport: FakeTransport) -> any LanguageModel {
        GeminiLanguageModel(apiKey: "test", model: "test", transport: transport)
    }

    private static func json(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }

    private static func reply(parts: [[String: Any]], streaming: Bool) throws -> FakeTransport.Reply {
        let body = try json(["candidates": [["content": ["role": "model", "parts": parts], "finishReason": "STOP"]]])
        return .init(chunks: [streaming ? "data: \(body)\n\n" : body])
    }

    private static func text(_ text: String, signature: String? = nil) -> [String: Any] {
        var part: [String: Any] = ["text": text]
        if let signature { part["thoughtSignature"] = signature }
        return part
    }

    private static var call: [String: Any] {
        ["functionCall": ["name": "getWeather", "args": ["city": "Paris"]], "thoughtSignature": "call-signature"]
    }

    /// One label for each part of each content, with its role and signature.
    private static func labels(_ body: Data) throws -> [String] {
        let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let contents = try #require(json["contents"] as? [[String: Any]])
        return contents.flatMap { content -> [String] in
            let role = content["role"] as? String ?? "?"
            return (content["parts"] as? [[String: Any]] ?? []).map { part in
                let signature = (part["thoughtSignature"] as? String).map { " #\($0)" } ?? ""
                if let text = part["text"] as? String { return "\(role): \(text)\(signature)" }
                if part["functionCall"] != nil { return "call\(signature)" }
                if part["functionResponse"] != nil { return "output" }
                return "other"
            }
        }
    }

    private static func ask(_ session: LanguageModelSession, _ prompt: String, streaming: Bool) async throws -> String {
        try await streaming ? session.streamResponse(to: prompt).collect().content : session.respond(to: prompt).content
    }

    enum Edit: CaseIterable, Sendable {
        case none, segmentsInPlace, rebuilt, removed

        func apply(to entries: [Transcript.Entry]) -> [Transcript.Entry] {
            entries.compactMap { entry in
                guard case .response(var response) = entry else { return entry }
                switch self {
                case .none:
                    return entry
                case .segmentsInPlace:
                    response.segments = [.text(.init(content: "REDACTED"))]
                    return .response(response)
                case .rebuilt:
                    return .response(.init(assetIDs: [], segments: [.text(.init(content: "REDACTED"))]))
                case .removed:
                    return nil
                }
            }
        }
    }

    private static func session(
        _ transport: FakeTransport,
        edit: Edit
    ) -> LanguageModelSession {
        LanguageModelSession(
            profile: LanguageModelSession.Profile { WeatherTool() }
                .model(model(transport))
                .historyTransform { edit.apply(to: $0) }
        )
    }

    static var cases: [(Edit, Bool)] {
        Edit.allCases.flatMap { edit in [(edit, false), (edit, true)] }
    }

    @Test(arguments: Self.cases)
    func aSignedAnswerFollowsItsEdits(edit: Edit, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text("SECRET", signature: "text-signature")], streaming: streaming),
            try Self.reply(parts: [Self.text("Bye.")], streaming: streaming),
        ])
        let session = Self.session(transport, edit: edit)
        _ = try await Self.ask(session, "Hello", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        let expected: [String] =
            switch edit {
            case .none: ["user: Hello", "model: SECRET #text-signature", "user: Thanks"]
            case .segmentsInPlace, .rebuilt: ["user: Hello", "model: REDACTED", "user: Thanks"]
            case .removed: ["user: Hello", "user: Thanks"]
            }
        #expect(try Self.labels(transport.bodies[1]) == expected)
    }

    /// The text parts of a tool round follow an edit to the response that shows them.
    /// A function call keeps its signature, because the transform didn't touch the call.
    @Test(arguments: Self.cases)
    func aToolRoundsTextFollowsEditsToItsResponse(edit: Edit, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(
                parts: [Self.text("SECRET.", signature: "round-signature"), Self.call],
                streaming: streaming
            ),
            try Self.reply(parts: [Self.text("Sunny.", signature: "text-signature")], streaming: streaming),
            try Self.reply(parts: [Self.text("Bye.")], streaming: streaming),
        ])
        let session = Self.session(transport, edit: edit)
        let first = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(first == "SECRET.Sunny.")
        // The round in flight has no response to edit yet.
        #expect(
            try Self.labels(transport.bodies[1]) == [
                "user: Weather?", "model: SECRET. #round-signature", "call #call-signature", "output",
            ]
        )
        let expected: [String] =
            switch edit {
            case .none:
                [
                    "user: Weather?", "model: SECRET. #round-signature", "call #call-signature", "output",
                    "model: Sunny. #text-signature", "user: Thanks",
                ]
            case .segmentsInPlace, .rebuilt:
                ["user: Weather?", "call #call-signature", "output", "model: REDACTED", "user: Thanks"]
            case .removed:
                ["user: Weather?", "call #call-signature", "output", "user: Thanks"]
            }
        #expect(try Self.labels(transport.bodies[2]) == expected)
    }

    @Test(arguments: [false, true])
    func anUnsignedAnswerAfterAToolRoundSendsItsTextOnce(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text("Checking."), Self.call], streaming: streaming),
            try Self.reply(parts: [Self.text("Sunny.")], streaming: streaming),
            try Self.reply(parts: [Self.text("Bye.")], streaming: streaming),
        ])
        let session = Self.session(transport, edit: .none)
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(
            try Self.labels(transport.bodies[2]) == [
                "user: Weather?", "model: Checking.", "call #call-signature", "output", "model: Sunny.", "user: Thanks",
            ]
        )
    }

    /// A delegate's message doesn't start a new turn,
    /// so the round before it keeps its text and the text's place on the next turn.
    @Test(arguments: [false, true])
    func aPendingMessageKeepsTheTextOfTheRoundBeforeIt(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text("Checking."), Self.call], streaming: streaming),
            try Self.reply(parts: [Self.text("Sunny.")], streaming: streaming),
            try Self.reply(parts: [Self.text("Bye.")], streaming: streaming),
        ])
        let session = Self.session(transport, edit: .none)
        session.toolExecutionDelegate = SteeringGeminiDelegate()
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(
            try Self.labels(transport.bodies[2]) == [
                "user: Weather?", "model: Checking.", "call #call-signature", "output", "user: Be brief.",
                "model: Sunny.", "user: Thanks",
            ]
        )
    }

    /// A saved transcript has the same metadata, whichever release wrote it.
    @Test(arguments: [false, true])
    func aSavedTranscriptReplaysItsPartsAndSignatures(edited: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text("SECRET.", signature: "round-signature"), Self.call], streaming: false),
            try Self.reply(parts: [Self.text("Sunny.", signature: "text-signature")], streaming: false),
            try Self.reply(parts: [Self.text("Bye.")], streaming: false),
        ])
        let source = LanguageModelSession(model: Self.model(transport), tools: [WeatherTool()])
        _ = try await source.respond(to: "Weather?")

        let saved = try JSONDecoder().decode(Transcript.self, from: try JSONEncoder().encode(source.transcript))
        let transcript = edited ? Transcript(entries: Edit.segmentsInPlace.apply(to: Array(saved))) : saved
        let session = LanguageModelSession(model: Self.model(transport), transcript: transcript)
        _ = try await session.respond(to: "Thanks")

        #expect(
            try Self.labels(transport.bodies[2])
                == (edited
                    ? ["user: Weather?", "call #call-signature", "output", "model: REDACTED", "user: Thanks"]
                    : [
                        "user: Weather?", "model: SECRET. #round-signature", "call #call-signature", "output",
                        "model: Sunny. #text-signature", "user: Thanks",
                    ])
        )
    }

    @Test(arguments: [false, true])
    func aStoppedToolRoundSendsItsTextOnce(streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(
                parts: [Self.text("Checking.", signature: "round-signature"), Self.call],
                streaming: streaming
            ),
            try Self.reply(parts: [Self.text("Bye.")], streaming: streaming),
        ])
        let session = LanguageModelSession(model: Self.model(transport), tools: [WeatherTool()])
        session.toolExecutionDelegate = StoppingGeminiDelegate()
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(
            try Self.labels(transport.bodies[1]) == [
                "user: Weather?", "model: Checking. #round-signature", "call #call-signature", "output",
                "user: Thanks",
            ]
        )
    }

    /// A stopped stream's response is the round's text. The other builders can't read Gemini's
    /// text parts, so they send that response, once.
    @Test
    func aStoppedStreamSendsItsTextOnceInOtherBuilders() async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text("Checking."), Self.call], streaming: true)
        ])
        let source = LanguageModelSession(model: Self.model(transport), tools: [WeatherTool()])
        source.toolExecutionDelegate = StoppingGeminiDelegate()
        _ = try await source.streamResponse(to: "Weather?").collect()

        let bye = #"{"id": "r", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Bye."}]}]}"#
        let chatBye = #"{"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Bye."}}]}"#
        let models: [(FakeTransport) -> any LanguageModel] = [
            { OpenAILanguageModel(apiKey: "t", model: "t", apiVariant: .chatCompletions, transport: $0) },
            { OpenAILanguageModel(apiKey: "t", model: "t", apiVariant: .responses, transport: $0) },
            {
                OpenResponsesLanguageModel(
                    baseURL: URL(string: "https://example.com/v1/")!,
                    apiKey: "t",
                    model: "t",
                    transport: $0
                )
            },
        ]
        for (index, model) in models.enumerated() {
            let next = FakeTransport(replies: [.init(chunks: [index == 0 ? chatBye : bye])])
            let session = LanguageModelSession(model: model(next), transcript: source.transcript)
            _ = try await session.respond(to: "Thanks")
            let body = String(decoding: next.bodies[0], as: UTF8.self)
            #expect(body.components(separatedBy: "Checking.").count - 1 == 1)
        }
    }

    /// A transform that empties a stopped stream's response removes the round's text too.
    @Test
    func emptyingAStoppedStreamsResponseRemovesItsText() async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text("SECRET."), Self.call], streaming: true),
            try Self.reply(parts: [Self.text("Bye.")], streaming: true),
        ])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { WeatherTool() }
                .model(Self.model(transport))
                .historyTransform { entries in
                    entries.map { entry in
                        guard case .response(var response) = entry else { return entry }
                        response.segments = [.text(.init(content: ""))]
                        return .response(response)
                    }
                }
        )
        session.toolExecutionDelegate = StoppingGeminiDelegate()
        _ = try await Self.ask(session, "Weather?", streaming: true)
        _ = try await Self.ask(session, "Thanks", streaming: true)

        #expect(!String(decoding: transport.bodies[1], as: UTF8.self).contains("SECRET."))
    }

    @Test(arguments: [false, true])
    func aSignedStructuredAnswerFollowsItsEdits(edited: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text(#"{"text":"SECRET"}"#, signature: "text-signature")], streaming: false),
            try Self.reply(parts: [Self.text("Bye.")], streaming: false),
        ])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile {}
                .model(Self.model(transport))
                .historyTransform { entries in
                    guard edited else { return entries }
                    return entries.map { entry in
                        guard case .response(var response) = entry else { return entry }
                        response.segments = [.text(.init(content: #"{"text":"REDACTED"}"#))]
                        return .response(response)
                    }
                }
        )
        _ = try await session.respond(to: "Hello", generating: GeminiGreeting.self)
        _ = try await session.respond(to: "Thanks")

        let labels = try Self.labels(transport.bodies[1])
        #expect(
            Array(labels.dropFirst()) == [
                edited ? #"model: {"text":"REDACTED"}"# : #"model: {"text":"SECRET"} #text-signature"#,
                "user: Thanks",
            ]
        )
    }

    /// The JSON that a stream emitted with its tool call is the stopped response's text.
    @Test(arguments: [false, true])
    func aStoppedStructuredStreamSendsItsResponseOnce(edited: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.reply(parts: [Self.text(#"{"text":"SECRET"}"#), Self.call], streaming: true),
            try Self.reply(parts: [Self.text("Bye.")], streaming: true),
        ])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { WeatherTool() }
                .model(Self.model(transport))
                .historyTransform { entries in
                    guard edited else { return entries }
                    return entries.map { entry in
                        guard case .response(var response) = entry else { return entry }
                        response.segments = [.text(.init(content: #"{"text":"REDACTED"}"#))]
                        return .response(response)
                    }
                }
        )
        session.toolExecutionDelegate = StoppingGeminiDelegate()
        _ = try await session.streamResponse(to: "Hello", generating: GeminiGreeting.self).collect()
        _ = try await Self.ask(session, "Thanks", streaming: true)

        let labels = try Self.labels(transport.bodies[1])
        #expect(
            Array(labels.dropFirst()) == [
                "call #call-signature",
                "output",
                edited ? #"model: {"text":"REDACTED"}"# : #"model: {"text":"SECRET"}"#,
                "user: Thanks",
            ]
        )
    }
}

/// Hands over one message after the first round of tool calls.
private final class SteeringGeminiDelegate: ToolExecutionDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var pending = [AnyLanguageModel.Prompt("Be brief.")]

    func pendingMessages(in session: LanguageModelSession) async -> [AnyLanguageModel.Prompt] {
        lock.withLock {
            defer { pending = [] }
            return pending
        }
    }
}

private struct StoppingGeminiDelegate: ToolExecutionDelegate {
    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        .stop
    }
}

@Generable
private struct GeminiGreeting {
    var text: String
}
