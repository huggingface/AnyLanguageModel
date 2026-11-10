import Foundation
import Testing

@testable import AnyLanguageModel

/// A `respond` that's cancelled during a tool round doesn't send the next request,
/// even when the transport doesn't stop for cancellation.
@Suite("Respond cancellation")
struct RespondCancellationTests {
    enum API: CaseIterable, Sendable {
        case chatCompletions, openAIResponses, openResponses, anthropic, gemini, ollama
    }

    private static func json(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }

    /// A model for `api` whose first reply calls `getWeather` and whose second answers.
    /// ``FakeTransport`` doesn't check for cancellation.
    private static func makeModel(_ api: API) throws -> (any LanguageModel, FakeTransport) {
        let arguments: [String: Any] = ["city": "Paris"]
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
                        "content": [["type": "tool_use", "id": "call_1", "name": "getWeather", "input": arguments]],
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
                try reply(["functionCall": ["name": "getWeather", "args": arguments]]),
                try reply(["text": "Bye."]),
            ])
            return (GeminiLanguageModel(apiKey: "test", model: "test", transport: transport), transport)
        case .ollama:
            func reply(_ message: [String: Any]) throws -> FakeTransport.Reply {
                .init(chunks: [
                    try json(["model": "m", "created_at": "2026-11-10T10:00:00.000Z", "message": message, "done": true])
                        + "\n"
                ])
            }
            let transport = FakeTransport(replies: [
                try reply([
                    "role": "assistant", "content": "",
                    "tool_calls": [
                        [
                            "id": "call_1", "type": "function",
                            "function": ["name": "getWeather", "arguments": arguments],
                        ]
                    ],
                ]),
                try reply(["role": "assistant", "content": "Bye."]),
            ])
            return (
                OllamaLanguageModel(baseURL: URL(string: "http://localhost:11434")!, model: "m", transport: transport),
                transport
            )
        }
    }

    /// A tool that runs until its task is cancelled and then returns as if it finished.
    struct ToolIgnoringCancellation: Tool {
        let name = "getWeather"
        let description = "Retrieve the latest weather information for a city"
        let started: Locked<Bool>

        @Generable
        struct Arguments {
            var city: String
        }

        func call(arguments: Arguments) async throws -> String {
            started.withLock { $0 = true }
            while !Task.isCancelled { await Task.yield() }
            return "Sunny."
        }
    }

    @Test(arguments: API.allCases)
    func cancellationDuringAToolRoundStopsTheNextRequest(api: API) async throws {
        let (model, transport) = try Self.makeModel(api)
        let started = Locked(false)
        let session = LanguageModelSession(model: model, tools: [ToolIgnoringCancellation(started: started)])
        let task = Task { try await session.respond(to: "Weather?") }
        while !started.withLock({ $0 }) { try await Task.sleep(for: .milliseconds(1)) }
        task.cancel()

        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(transport.bodies.count == 1)
    }
}
