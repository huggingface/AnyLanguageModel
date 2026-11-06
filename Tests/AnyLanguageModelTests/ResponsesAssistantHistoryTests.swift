import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

/// What the OpenAI Responses API and Open Responses builders send of an earlier conversation.
///
/// Each assertion reads `input` as a list of labels, in order, so a missing, repeated,
/// or misplaced piece of text changes the list.
@Suite("Responses assistant history")
struct ResponsesAssistantHistoryTests {
    enum Builder: CaseIterable, Sendable {
        case openAI, openResponses

        func makeModel(_ transport: FakeTransport) -> any LanguageModel {
            switch self {
            case .openAI:
                OpenAILanguageModel(apiKey: "test", model: "test", apiVariant: .responses, transport: transport)
            case .openResponses:
                OpenResponsesLanguageModel(
                    baseURL: URL(string: "https://example.com/v1/")!,
                    apiKey: "test",
                    model: "test",
                    transport: transport
                )
            }
        }
    }

    // MARK: Replies

    /// A reply with these output items, as a stream or as one response.
    private static func reply(output: [[String: Any]], text: String, streaming: Bool) throws -> FakeTransport.Reply {
        let response: [String: Any] = ["id": "resp_1", "output": output]
        func json(_ object: [String: Any]) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
        guard streaming else { return .init(chunks: [try json(response)]) }
        var events: [[String: Any]] = []
        if !text.isEmpty { events.append(["type": "response.output_text.delta", "delta": text]) }
        events.append(["type": "response.completed", "response": response])
        return .init(chunks: try events.map { "data: \(try json($0))\n\n" })
    }

    private static func message(_ text: String) -> [String: Any] {
        [
            "type": "message", "id": "msg_server", "role": "assistant", "status": "completed",
            "content": [["type": "output_text", "text": text, "annotations": []]],
        ]
    }

    private static func textReply(_ text: String, streaming: Bool) throws -> FakeTransport.Reply {
        try reply(output: [message(text)], text: text, streaming: streaming)
    }

    private static func toolReply(
        _ text: String = "",
        reasoning: Bool = false,
        streaming: Bool
    ) throws -> FakeTransport.Reply {
        var output: [[String: Any]] = []
        if reasoning {
            output.append(["type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"])
        }
        if !text.isEmpty { output.append(message(text)) }
        output.append([
            "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "getWeather",
            "arguments": #"{"city":"Paris"}"#,
        ])
        return try reply(output: output, text: text, streaming: streaming)
    }

    // MARK: Reading requests

    private static func input(_ body: Data) throws -> [[String: Any]] {
        let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        return try #require(json["input"] as? [[String: Any]])
    }

    /// The text of a message item, in either content form.
    private static func text(of item: [String: Any]) -> String {
        if let text = item["content"] as? String { return text }
        return (item["content"] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }.joined()
    }

    private static func labels(_ body: Data) throws -> [String] {
        try input(body).map { item in
            switch item["type"] as? String {
            case "message": "\(item["role"] as? String ?? "?"): \(text(of: item))"
            case "function_call": "call: \(item["call_id"] as? String ?? "?")"
            case "function_call_output": "output: \(item["call_id"] as? String ?? "?")"
            case "reasoning": "reasoning"
            default: "other: \(item["type"] as? String ?? "?")"
            }
        }
    }

    private static func ask(_ session: LanguageModelSession, _ prompt: String, streaming: Bool) async throws -> String {
        try await streaming ? session.streamResponse(to: prompt).collect().content : session.respond(to: prompt).content
    }

    private static let toolTurn = [
        "user: Weather?", "assistant: Checking.", "call: call_1", "output: call_1", "assistant: Sunny.",
    ]

    // MARK: Turns

    @Test(arguments: Builder.allCases, [false, true])
    func aSecondTurnContainsTheFirstAnswer(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.textReply("First.", streaming: streaming),
            try Self.textReply("Second.", streaming: streaming),
            try Self.textReply("Third.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport))
        _ = try await Self.ask(session, "One", streaming: streaming)
        _ = try await Self.ask(session, "Two", streaming: streaming)
        _ = try await Self.ask(session, "Three", streaming: streaming)

        #expect(try Self.labels(transport.bodies[1]) == ["user: One", "assistant: First.", "user: Two"])
        #expect(
            try Self.labels(transport.bodies[2]) == [
                "user: One", "assistant: First.", "user: Two", "assistant: Second.", "user: Three",
            ]
        )
    }

    @Test func openAISendsAnEasyInputMessage() async throws {
        let transport = FakeTransport(replies: [
            try Self.textReply("First.", streaming: false),
            try Self.textReply("Second.", streaming: false),
        ])
        let session = LanguageModelSession(model: Builder.openAI.makeModel(transport))
        _ = try await session.respond(to: "One")
        _ = try await session.respond(to: "Two")

        let item = try Self.input(transport.bodies[1])[1]
        #expect(item.count == 3)
        #expect(item["type"] as? String == "message")
        #expect(item["role"] as? String == "assistant")
        #expect(item["content"] as? String == "First.")
    }

    @Test func openResponsesSendsAnAssistantMessageItemWithAnIDAndStatus() async throws {
        let transport = FakeTransport(replies: [
            try Self.textReply("First.", streaming: false),
            try Self.textReply("Second.", streaming: false),
            try Self.textReply("Third.", streaming: false),
        ])
        let session = LanguageModelSession(model: Builder.openResponses.makeModel(transport))
        _ = try await session.respond(to: "One")
        _ = try await session.respond(to: "Two")
        _ = try await session.respond(to: "Three")

        let item = try Self.input(transport.bodies[1])[1]
        #expect(item.count == 5)
        #expect(item["type"] as? String == "message")
        #expect(item["role"] as? String == "assistant")
        #expect(item["status"] as? String == "completed")
        #expect((item["id"] as? String)?.hasPrefix("msg_") == true)
        let content = try #require(item["content"] as? [[String: Any]])
        try #require(content.count == 1)
        #expect(content[0]["type"] as? String == "output_text")
        #expect(content[0]["text"] as? String == "First.")
        #expect((content[0]["annotations"] as? [Any])?.isEmpty == true)
        // A later request sends the same ID, so its prefix stays the same for a prompt cache.
        #expect(try Self.input(transport.bodies[2])[1]["id"] as? String == item["id"] as? String)
    }

    // MARK: Tool rounds

    @Test(arguments: Builder.allCases, [false, true])
    func aToolRoundsTextIsSentOnceInEachRequest(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", streaming: streaming),
            try Self.textReply("Sunny.", streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        let first = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(first == "Checking.Sunny.")
        #expect(
            try Self.labels(transport.bodies[1]) == [
                "user: Weather?", "assistant: Checking.", "call: call_1", "output: call_1",
            ]
        )
        #expect(try Self.labels(transport.bodies[2]) == Self.toolTurn + ["user: Thanks"])
    }

    @Test(arguments: Builder.allCases, [false, true])
    func aLaterTurnAfterAToolRoundWithReasoningSendsTextOnce(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", reasoning: true, streaming: streaming),
            try Self.textReply("Sunny.", streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        // The round in flight sends the server's items as they came, reasoning included.
        #expect(
            try Self.labels(transport.bodies[1]) == [
                "user: Weather?", "reasoning", "assistant: Checking.", "call: call_1", "output: call_1",
            ]
        )
        // The transcript keeps no Responses reasoning, so a later turn sends none,
        // and its calls and messages have no server item IDs that reasoning must go with.
        #expect(try Self.labels(transport.bodies[2]) == Self.toolTurn + ["user: Thanks"])
        for item in try Self.input(transport.bodies[2]) {
            #expect((item["id"] as? String)?.hasPrefix("fc_") != true)
            #expect(item["id"] as? String != "msg_server")
        }
    }

    @Test(arguments: Builder.allCases, [false, true])
    func aToolRoundWithoutTextSendsNoEmptyMessage(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply(streaming: streaming),
            try Self.textReply("Sunny.", streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(
            try Self.labels(transport.bodies[2]) == [
                "user: Weather?", "call: call_1", "output: call_1", "assistant: Sunny.", "user: Thanks",
            ]
        )
    }

    @Test(arguments: Builder.allCases, [false, true])
    func aStoppedToolRoundSendsItsTextOnce(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        session.toolExecutionDelegate = StoppingDelegate()
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(
            try Self.labels(transport.bodies[1]) == [
                "user: Weather?", "assistant: Checking.", "call: call_1", "user: Thanks",
            ]
        )
    }

    // MARK: Structured responses

    @Test(arguments: Builder.allCases, [false, true])
    func aStructuredResponseIsSentOnceAsItsJSON(builder: Builder, streaming: Bool) async throws {
        let json = #"{"text":"Hi"}"#
        let transport = FakeTransport(replies: [
            try Self.textReply(json, streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport))
        let greeting =
            try await streaming
            ? session.streamResponse(to: "Hello", generating: Greeting.self).collect().content
            : session.respond(to: "Hello", generating: Greeting.self).content
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(greeting.text == "Hi")
        let labels = try Self.labels(transport.bodies[1])
        #expect(labels.first?.hasPrefix("user: Hello") == true)
        #expect(Array(labels.dropFirst()) == ["assistant: \(json)", "user: Thanks"])
    }

    @Test(arguments: Builder.allCases, [false, true])
    func aStructuredResponseAfterAToolRoundSendsEachRoundsTextOnce(builder: Builder, streaming: Bool) async throws {
        let json = #"{"text":"Hi"}"#
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", streaming: streaming),
            try Self.textReply(json, streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        _ =
            try await streaming
            ? session.streamResponse(to: "Hello", generating: Greeting.self).collect().content
            : session.respond(to: "Hello", generating: Greeting.self).content
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        let labels = try Self.labels(transport.bodies[2])
        #expect(
            Array(labels.dropFirst()) == [
                "assistant: Checking.", "call: call_1", "output: call_1", "assistant: \(json)", "user: Thanks",
            ]
        )
    }

    // MARK: Dynamic profiles

    @Test(arguments: Builder.allCases, [false, true])
    func aHistoryTransformKeepsEachTextOnce(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", streaming: streaming),
            try Self.textReply("Sunny.", streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { WeatherTool() }
                .model(builder.makeModel(transport))
                .historyTransform { $0 }
        )
        _ = try await Self.ask(session, "Weather?", streaming: streaming)
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(
            try Self.labels(transport.bodies[1]) == [
                "user: Weather?", "assistant: Checking.", "call: call_1", "output: call_1",
            ]
        )
        #expect(try Self.labels(transport.bodies[2]) == Self.toolTurn + ["user: Thanks"])
    }

    @Test(arguments: Builder.allCases, [false, true])
    func aHistoryTransformDecidesWhichAnswersAreSent(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.textReply("First.", streaming: streaming),
            try Self.textReply("Second.", streaming: streaming),
        ])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile {}
                .model(builder.makeModel(transport))
                .historyTransform { entries in
                    entries.filter { if case .response = $0 { false } else { true } }
                }
        )
        _ = try await Self.ask(session, "One", streaming: streaming)
        _ = try await Self.ask(session, "Two", streaming: streaming)

        #expect(try Self.labels(transport.bodies[1]) == ["user: One", "user: Two"])
    }

    /// A profile whose tool hands the rest of the turn to the second model.
    private struct HandoffProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
        let useSecond: Locked<Bool>
        let first: any LanguageModel
        let second: any LanguageModel

        var body: some LanguageModelSession.DynamicProfile {
            LanguageModelSession.Profile { HandoffTool(useSecond: useSecond) }
                .model(useSecond.withLock { $0 } ? second : first)
        }
    }

    private struct HandoffTool: Tool {
        let name = "getWeather"
        let description = "Get the weather and hand off"
        let useSecond: Locked<Bool>

        func call(arguments: WeatherTool.Arguments) async throws -> String {
            useSecond.withLock { $0 = true }
            return "Sunny"
        }
    }

    @Test(arguments: Builder.allCases, Builder.allCases)
    func aHandoffDuringATurnSendsEachTextOnce(from: Builder, to: Builder) async throws {
        for streaming in [false, true] {
            let firstTransport = FakeTransport(replies: [try Self.toolReply("Checking.", streaming: streaming)])
            let secondTransport = FakeTransport(replies: [
                try Self.textReply("Sunny.", streaming: streaming),
                try Self.textReply("Bye.", streaming: streaming),
            ])
            let session = LanguageModelSession(
                profile: HandoffProfile(
                    useSecond: Locked(false),
                    first: from.makeModel(firstTransport),
                    second: to.makeModel(secondTransport)
                )
            )
            let first = try await Self.ask(session, "Weather?", streaming: streaming)
            _ = try await Self.ask(session, "Thanks", streaming: streaming)

            #expect(first == "Checking.Sunny.")
            #expect(
                try Self.labels(secondTransport.bodies[0]) == [
                    "user: Weather?", "assistant: Checking.", "call: call_1", "output: call_1",
                ]
            )
            #expect(try Self.labels(secondTransport.bodies[1]) == Self.toolTurn + ["user: Thanks"])
        }
    }

    private static func chatModel(_ transport: FakeTransport) -> any LanguageModel {
        OpenAILanguageModel(apiKey: "test", model: "test", apiVariant: .chatCompletions, transport: transport)
    }

    private static func chatToolReply(_ text: String) -> FakeTransport.Reply {
        .init(chunks: [
            #"{"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": "\#(text)", "reasoning_content": "Hmm.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "getWeather", "arguments": "{\"city\": \"Paris\"}"}}]}}]}"#
        ])
    }

    private static func chatTextReply(_ text: String) -> FakeTransport.Reply {
        .init(chunks: [
            #"{"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": "\#(text)"}}]}"#
        ])
    }

    /// A Chat Completions transcript whose tool round has text and reasoning.
    private static func chatTranscript() async throws -> Transcript {
        let transport = FakeTransport(replies: [chatToolReply("Checking."), chatTextReply("Sunny.")])
        let session = LanguageModelSession(
            model: OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                apiVariant: .chatCompletions,
                transport: transport
            ),
            tools: [WeatherTool()]
        )
        _ = try await session.respond(to: "Weather?")
        return session.transcript
    }

    @Test(arguments: Builder.allCases)
    func aHandoffFromChatCompletionsDuringATurnSendsEachTextOnce(to: Builder) async throws {
        let chat = FakeTransport(replies: [Self.chatToolReply("Checking.")])
        let transport = FakeTransport(replies: [
            try Self.textReply("Sunny.", streaming: false),
            try Self.textReply("Bye.", streaming: false),
        ])
        let session = LanguageModelSession(
            profile: HandoffProfile(
                useSecond: Locked(false),
                first: OpenAILanguageModel(
                    apiKey: "test",
                    model: "test",
                    apiVariant: .chatCompletions,
                    transport: chat
                ),
                second: to.makeModel(transport)
            )
        )
        _ = try await session.respond(to: "Weather?")
        _ = try await session.respond(to: "Thanks")

        #expect(try Self.labels(transport.bodies[1]) == Self.toolTurn + ["user: Thanks"])
    }

    /// A handoff between Chat Completions models replays the first model's text with its calls only.
    @Test func aHandoffBetweenChatCompletionsModelsSendsEachTextOnce() async throws {
        let first = FakeTransport(replies: [Self.chatToolReply("Checking.")])
        let second = FakeTransport(replies: [Self.chatTextReply("Sunny."), Self.chatTextReply("Bye.")])
        let session = LanguageModelSession(
            profile: HandoffProfile(
                useSecond: Locked(false),
                first: Self.chatModel(first),
                second: Self.chatModel(second)
            )
        )
        _ = try await session.respond(to: "Weather?")
        _ = try await session.respond(to: "Thanks")

        let body = try #require(try JSONSerialization.jsonObject(with: second.bodies[1]) as? [String: Any])
        let messages = try #require(body["messages"] as? [[String: Any]])
        let assistant = messages.filter { $0["role"] as? String == "assistant" }.map(Self.text(of:))
        #expect(assistant == ["Checking.", "Sunny."])
    }

    @Test(arguments: Builder.allCases)
    func aChatCompletionsTranscriptSendsEachTextOnce(builder: Builder) async throws {
        let transport = FakeTransport(replies: [try Self.textReply("Bye.", streaming: false)])
        let session = LanguageModelSession(
            model: builder.makeModel(transport),
            transcript: try await Self.chatTranscript()
        )
        _ = try await session.respond(to: "Thanks")

        #expect(try Self.labels(transport.bodies[0]) == Self.toolTurn + ["user: Thanks"])
        let json = String(decoding: transport.bodies[0], as: UTF8.self)
        #expect(!json.contains("reasoning_content"))
    }

    @Test(arguments: Builder.allCases)
    func chatCompletionsSendsAResponsesTranscriptsTextOnce(builder: Builder) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", streaming: false),
            try Self.textReply("Sunny.", streaming: false),
        ])
        let source = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        _ = try await source.respond(to: "Weather?")

        let chat = FakeTransport(replies: [Self.chatTextReply("Bye.")])
        let session = LanguageModelSession(
            model: Self.chatModel(chat),
            transcript: source.transcript
        )
        _ = try await session.respond(to: "Thanks")

        let body = try #require(try JSONSerialization.jsonObject(with: chat.bodies[0]) as? [String: Any])
        let messages = try #require(body["messages"] as? [[String: Any]])
        let assistant = messages.filter { $0["role"] as? String == "assistant" }.map(Self.text(of:))
        #expect(assistant == ["Checking.", "Sunny."])
    }

    // MARK: Restored transcripts

    @Test(arguments: Builder.allCases, [false, true])
    func aSavedTranscriptReplaysEachTextOnce(builder: Builder, streaming: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.toolReply("Checking.", streaming: streaming),
            try Self.textReply("Sunny.", streaming: streaming),
            try Self.textReply("Bye.", streaming: streaming),
        ])
        let source = LanguageModelSession(model: builder.makeModel(transport), tools: [WeatherTool()])
        _ = try await Self.ask(source, "Weather?", streaming: streaming)

        let saved = try JSONEncoder().encode(source.transcript)
        let session = LanguageModelSession(
            model: builder.makeModel(transport),
            transcript: try JSONDecoder().decode(Transcript.self, from: saved)
        )
        _ = try await Self.ask(session, "Thanks", streaming: streaming)

        #expect(try Self.labels(transport.bodies[2]) == Self.toolTurn + ["user: Thanks"])
    }

    /// A transcript that an earlier release saved has all of a turn's text in its response.
    @Test(arguments: Builder.allCases)
    func aTranscriptWithoutRoundTextSendsTheResponseOnce(builder: Builder) async throws {
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "getWeather",
            arguments: try GeneratedContent(json: #"{"city":"Paris"}"#)
        )
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Weather?"))])),
            .toolCalls(.init([call])),
            .toolOutput(.init(id: "call_1", toolName: "getWeather", segments: [.text(.init(content: "Sunny"))])),
            .response(.init(assetIDs: [], segments: [.text(.init(content: "Checking.Sunny."))])),
        ])
        let transport = FakeTransport(replies: [try Self.textReply("Bye.", streaming: false)])
        let session = LanguageModelSession(model: builder.makeModel(transport), transcript: transcript)
        _ = try await session.respond(to: "Thanks")

        #expect(
            try Self.labels(transport.bodies[0]) == [
                "user: Weather?", "call: call_1", "output: call_1", "assistant: Checking.Sunny.", "user: Thanks",
            ]
        )
    }

    // MARK: Non-text content

    @Test(arguments: Builder.allCases)
    func nonTextSegmentsBecomeText(builder: Builder) async throws {
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Draw."))])),
            .response(
                .init(
                    assetIDs: [],
                    segments: [
                        .text(.init(content: "Here:")),
                        .image(.init(data: testImageData, mimeType: "image/png")),
                        .structure(.init(source: "Greeting", content: try GeneratedContent(json: #"{"text":"Hi"}"#))),
                    ]
                )
            ),
            .prompt(.init(segments: [.text(.init(content: "Again."))])),
            .response(.init(assetIDs: [], segments: [.image(.init(url: testImageURL))])),
        ])
        let transport = FakeTransport(replies: [try Self.textReply("Bye.", streaming: false)])
        let session = LanguageModelSession(model: builder.makeModel(transport), transcript: transcript)
        _ = try await session.respond(to: "Thanks")

        #expect(
            try Self.labels(transport.bodies[0]) == [
                "user: Draw.",
                "assistant: Here:\n<image>\n{\"text\":\"Hi\"}",
                "user: Again.",
                "assistant: <image>",
                "user: Thanks",
            ]
        )
        let json = String(decoding: transport.bodies[0], as: UTF8.self)
        #expect(!json.contains("input_image"))
        #expect(!json.contains(testImageData.base64EncodedString()))
    }

    // MARK: Server state

    @Test(arguments: [false, true])
    func storeDoesNotChangeTheHistorySent(store: Bool) async throws {
        let transport = FakeTransport(replies: [
            try Self.textReply("First.", streaming: false),
            try Self.textReply("Second.", streaming: false),
        ])
        var options = GenerationOptions()
        options[custom: OpenAILanguageModel.self] = .init(store: store)
        let session = LanguageModelSession(model: Builder.openAI.makeModel(transport))
        _ = try await session.respond(to: "One", options: options)
        _ = try await session.respond(to: "Two", options: options)

        #expect(try Self.labels(transport.bodies[1]) == ["user: One", "assistant: First.", "user: Two"])
        let body = try #require(try JSONSerialization.jsonObject(with: transport.bodies[1]) as? [String: Any])
        #expect(body["store"] as? Bool == store)
        #expect(body["previous_response_id"] == nil)
    }
}

private struct StoppingDelegate: ToolExecutionDelegate {
    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        .stop
    }
}

@Generable
private struct Greeting {
    var text: String
}
