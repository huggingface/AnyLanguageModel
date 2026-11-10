import Foundation
import Testing

@testable import AnyLanguageModel

/// After a ``ToolExecutionDelegate`` stops a round, the next request has an output for each call,
/// because each API rejects a tool call that no output follows.
@Suite("Stopped tool call replay")
struct StoppedToolCallReplayTests {
    enum API: CaseIterable, Sendable {
        case chatCompletions, openAIResponses, openResponses, anthropic, gemini, ollama
    }

    private static func json(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }

    private static func ollamaReply(_ message: [String: Any]) throws -> FakeTransport.Reply {
        .init(chunks: [
            try json(["model": "m", "created_at": "2026-11-10T10:00:00.000Z", "message": message, "done": true]) + "\n"
        ])
    }

    /// A model for `api` whose first reply calls `getWeather` and whose second answers.
    private static func makeModel(_ api: API) throws -> (any LanguageModel, FakeTransport) {
        let call: [String: Any] = ["city": "Paris"]
        switch api {
        case .chatCompletions:
            let transport = FakeTransport(replies: [
                .init(chunks: [
                    #"{"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "getWeather", "arguments": "{\"city\": \"Paris\"}"}}]}}]}"#
                ]),
                .init(chunks: [
                    #"{"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Bye."}}]}"#
                ]),
            ])
            return (
                OpenAILanguageModel(apiKey: "test", model: "test", apiVariant: .chatCompletions, transport: transport),
                transport
            )
        case .openAIResponses, .openResponses:
            let functionCall: [String: Any] = [
                "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "getWeather",
                "arguments": #"{"city":"Paris"}"#,
            ]
            let message: [String: Any] = [
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [["type": "output_text", "text": "Bye.", "annotations": []]],
            ]
            let transport = FakeTransport(replies: [
                .init(chunks: [try json(["id": "resp_1", "output": [functionCall]])]),
                .init(chunks: [try json(["id": "resp_2", "output": [message]])]),
            ])
            let model: any LanguageModel =
                api == .openAIResponses
                ? OpenAILanguageModel(apiKey: "test", model: "test", apiVariant: .responses, transport: transport)
                : OpenResponsesLanguageModel(
                    baseURL: URL(string: "https://example.com/v1/")!,
                    apiKey: "test",
                    model: "test",
                    transport: transport
                )
            return (model, transport)
        case .anthropic:
            let usage: [String: Any] = ["input_tokens": 1, "output_tokens": 1]
            let transport = FakeTransport(replies: [
                .init(chunks: [
                    try json([
                        "id": "msg_1", "type": "message", "role": "assistant", "model": "test",
                        "content": [["type": "tool_use", "id": "call_1", "name": "getWeather", "input": call]],
                        "stop_reason": "tool_use", "usage": usage,
                    ])
                ]),
                .init(chunks: [
                    try json([
                        "id": "msg_2", "type": "message", "role": "assistant", "model": "test",
                        "content": [["type": "text", "text": "Bye."]], "stop_reason": "end_turn", "usage": usage,
                    ])
                ]),
            ])
            return (AnthropicLanguageModel(apiKey: "test", model: "test", transport: transport), transport)
        case .gemini:
            func reply(_ part: [String: Any]) throws -> FakeTransport.Reply {
                .init(chunks: [try json(["candidates": [["content": ["role": "model", "parts": [part]]]]])])
            }
            let transport = FakeTransport(replies: [
                try reply(["functionCall": ["name": "getWeather", "args": call]]),
                try reply(["text": "Bye."]),
            ])
            return (GeminiLanguageModel(apiKey: "test", model: "test", transport: transport), transport)
        case .ollama:
            let transport = FakeTransport(replies: [
                try ollamaReply([
                    "role": "assistant", "content": "",
                    "tool_calls": [
                        ["id": "call_1", "type": "function", "function": ["name": "getWeather", "arguments": call]]
                    ],
                ]),
                try ollamaReply(["role": "assistant", "content": "Bye."]),
            ])
            return (
                OllamaLanguageModel(baseURL: URL(string: "http://localhost:11434")!, model: "m", transport: transport),
                transport
            )
        }
    }

    /// Whether `value` has the text of a call that didn't run, at any depth.
    private static func hasNotExecutedText(_ value: Any) -> Bool {
        let data = (try? JSONSerialization.data(withJSONObject: ["value": value])) ?? Data()
        return String(decoding: data, as: UTF8.self).contains(notExecutedToolOutputText)
    }

    /// The request's history as labels: `user` and `assistant` text, `call`, and `output`,
    /// with `output (not executed)` for an output that says the call didn't run.
    /// Messages with empty text, such as the stopped turn's empty response, aren't included.
    private static func labels(_ body: Data, api: API) throws -> [String] {
        try allLabels(body, api: api).filter { !$0.hasSuffix(": ") }
    }

    private static func allLabels(_ body: Data, api: API) throws -> [String] {
        let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        func output(_ item: Any) -> String {
            hasNotExecutedText(item) ? "output (not executed)" : "output"
        }
        func text(_ content: Any?) -> String {
            if let text = content as? String { return text }
            return (content as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }.joined()
        }
        switch api {
        case .chatCompletions, .ollama:
            let messages = try #require(json["messages"] as? [[String: Any]])
            return messages.flatMap { message -> [String] in
                let role = message["role"] as? String ?? "?"
                if role == "tool" { return [output(message)] }
                if let calls = message["tool_calls"] as? [Any], !calls.isEmpty { return calls.map { _ in "call" } }
                return ["\(role): \(text(message["content"]))"]
            }
        case .openAIResponses, .openResponses:
            let input = try #require(json["input"] as? [[String: Any]])
            return input.map { item in
                switch item["type"] as? String {
                case "function_call": "call"
                case "function_call_output": output(item)
                default: "\(item["role"] as? String ?? "?"): \(text(item["content"]))"
                }
            }
        case .anthropic:
            let messages = try #require(json["messages"] as? [[String: Any]])
            return messages.flatMap { message -> [String] in
                let role = message["role"] as? String ?? "?"
                return (message["content"] as? [[String: Any]] ?? []).map { block in
                    switch block["type"] as? String {
                    case "tool_use": "call"
                    case "tool_result": output(block)
                    default: "\(role): \(block["text"] as? String ?? "")"
                    }
                }
            }
        case .gemini:
            let contents = try #require(json["contents"] as? [[String: Any]])
            return contents.flatMap { content -> [String] in
                let role = content["role"] as? String ?? "?"
                return (content["parts"] as? [[String: Any]] ?? []).map { part in
                    if part["functionCall"] != nil { return "call" }
                    if let response = part["functionResponse"] { return output(response) }
                    return "\(role): \(part["text"] as? String ?? "")"
                }
            }
        }
    }

    @Test(arguments: API.allCases)
    func theNextRequestHasAnOutputForEachStoppedCall(api: API) async throws {
        let (model, transport) = try Self.makeModel(api)
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])
        session.toolExecutionDelegate = StopToolExecutionDelegate()
        _ = try await session.respond(to: "Weather?")
        _ = try await session.respond(to: "Thanks")

        try #require(transport.bodies.count == 2)
        #expect(
            try Self.labels(transport.bodies[1], api: api) == [
                "user: Weather?", "call", "output (not executed)", "user: Thanks",
            ]
        )
        // The transcript keeps the stopped calls as they were.
        #expect(
            !session.transcript.contains { entry in
                if case .toolOutput = entry { true } else { false }
            }
        )
    }

    @Test func outputsAreAddedOnlyForCallsWithout() {
        let calls = Transcript.ToolCalls(
            id: "round",
            [
                .init(id: "a", toolName: "getWeather", arguments: GeneratedContent(properties: [:])),
                .init(id: "b", toolName: "getWeather", arguments: GeneratedContent(properties: [:])),
            ]
        )
        let ran = Transcript.ToolOutput(id: "a", toolName: "getWeather", segments: [.text(.init(content: "Sunny."))])
        let prompt = Transcript.Prompt(segments: [.text(.init(content: "Thanks"))])
        let entries = addingNotExecutedToolOutputs(to: [.toolCalls(calls), .toolOutput(ran), .prompt(prompt)])

        #expect(entries.count == 4)
        #expect(entries[1] == .toolOutput(ran))
        guard case .toolOutput(let added) = entries[2] else {
            Issue.record("Expected an output for call b")
            return
        }
        #expect(added.id == "b")
        #expect(added.toolName == "getWeather")
        #expect(entries[3] == .prompt(prompt))
    }
}
