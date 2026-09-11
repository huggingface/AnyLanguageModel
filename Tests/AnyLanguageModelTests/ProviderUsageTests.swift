import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Provider token usage", .serialized)
    struct ProviderUsageTests {
        enum Provider: CaseIterable, Sendable {
            case chat, responses, openResponses, anthropic, gemini, ollama

            func makeSession(tools: [any Tool] = []) -> LanguageModelSession {
                let http = UsageURLProtocol.makeSession()
                let model: any LanguageModel
                switch self {
                case .chat, .responses:
                    model = OpenAILanguageModel(
                        apiKey: "test",
                        model: "test",
                        apiVariant: self == .chat ? .chatCompletions : .responses,
                        session: http
                    )
                case .openResponses:
                    model = OpenResponsesLanguageModel(
                        baseURL: URL(string: "https://example.com/v1")!,
                        apiKey: "test",
                        model: "test",
                        session: http
                    )
                case .anthropic:
                    model = AnthropicLanguageModel(apiKey: "test", model: "test", session: http)
                case .gemini:
                    model = GeminiLanguageModel(apiKey: "test", model: "test", session: http)
                case .ollama:
                    model = OllamaLanguageModel(model: "test", session: http)
                }
                return LanguageModelSession(model: model, tools: tools)
            }

            var counts: [String: Any] {
                switch self {
                case .chat:
                    return [
                        "prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": ["cached_tokens": 25],
                        "completion_tokens_details": ["reasoning_tokens": 5],
                    ]
                case .responses, .openResponses:
                    return [
                        "input_tokens": 100, "output_tokens": 20, "input_tokens_details": ["cached_tokens": 25],
                        "output_tokens_details": ["reasoning_tokens": 5],
                    ]
                case .anthropic:
                    return [
                        "input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 25,
                        "cache_creation_input_tokens": 10,
                    ]
                case .gemini:
                    return ["promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5]
                case .ollama:
                    return ["prompt_eval_count": 100, "eval_count": 20]
                }
            }

            var expected: LanguageModelSession.Usage {
                .init(
                    input: .init(totalTokenCount: 100, cachedTokenCount: self == .gemini || self == .ollama ? 0 : 25),
                    output: .init(
                        totalTokenCount: 20,
                        reasoningTokenCount: self == .anthropic || self == .ollama ? 0 : 5
                    )
                )
            }

            func response(text: String = "Hello", counts: [String: Any]? = nil, tool: Bool = false) -> [String: Any] {
                var result: [String: Any]
                switch self {
                case .chat:
                    var message: [String: Any] = ["role": "assistant", "content": text]
                    if tool {
                        message["tool_calls"] = [
                            [
                                "id": "call_1", "type": "function",
                                "function": ["name": "getWeather", "arguments": "{\"city\":\"Paris\"}"],
                            ]
                        ]
                    }
                    result = ["id": "test", "choices": [["message": message]]]
                case .responses, .openResponses:
                    let output: [[String: Any]] =
                        tool
                        ? [
                            [
                                "type": "function_call", "call_id": "call_1", "name": "getWeather",
                                "arguments": "{\"city\":\"Paris\"}",
                            ]
                        ]
                        : [["type": "message", "content": [["type": "output_text", "text": text]]]]
                    result = ["id": "test", "output": output]
                case .anthropic:
                    let content: [[String: Any]] =
                        tool
                        ? [["type": "tool_use", "id": "call_1", "name": "getWeather", "input": ["city": "Paris"]]]
                        : [["type": "text", "text": text]]
                    result = [
                        "id": "test", "type": "message", "role": "assistant", "model": "test", "content": content,
                    ]
                case .gemini:
                    let part: [String: Any] =
                        tool
                        ? ["functionCall": ["name": "getWeather", "args": ["city": "Paris"]]]
                        : ["text": text]
                    result = ["candidates": [["content": ["role": "model", "parts": [part]]]]]
                case .ollama:
                    var message: [String: Any] = ["role": "assistant", "content": text]
                    if tool {
                        message["tool_calls"] = [["function": ["name": "getWeather", "arguments": ["city": "Paris"]]]]
                    }
                    result = [
                        "model": "test", "created_at": "2026-09-11T00:00:00.000Z", "message": message, "done": true,
                    ]
                }
                if let counts {
                    if self == .ollama {
                        result.merge(counts) { _, new in new }
                    } else {
                        result[self == .gemini ? "usageMetadata" : "usage"] = counts
                    }
                }
                return result
            }

            func stream(text: String = "Hello", includeUsage: Bool = true) throws -> String {
                var events: [[String: Any]]
                switch self {
                case .chat:
                    events = [
                        ["id": "test", "choices": [["delta": ["content": text]]]],
                        ["id": "test", "choices": [["delta": [:], "finish_reason": "stop"]]],
                    ]
                    if includeUsage { events.append(["id": "test", "choices": [], "usage": counts]) }
                case .responses, .openResponses:
                    events = [
                        ["type": "response.output_text.delta", "delta": text],
                        [
                            "type": "response.completed",
                            "response": response(text: text, counts: includeUsage ? counts : nil),
                        ],
                    ]
                case .anthropic:
                    var startCounts = counts
                    startCounts["output_tokens"] = 0
                    events = [
                        [
                            "type": "message_start",
                            "message": response(text: "", counts: includeUsage ? startCounts : nil),
                        ],
                        ["type": "content_block_delta", "index": 0, "delta": ["type": "text_delta", "text": text]],
                    ]
                    for output in [7, 20] {
                        var delta: [String: Any] = ["type": "message_delta", "delta": ["stop_reason": "end_turn"]]
                        if includeUsage { delta["usage"] = ["output_tokens": output] }
                        events.append(delta)
                    }
                    events.append(["type": "message_stop"])
                case .gemini:
                    var early = counts
                    early["candidatesTokenCount"] = 7
                    events = [response(text: text, counts: includeUsage ? early : nil)]
                    if includeUsage { events.append(["usageMetadata": counts]) }
                case .ollama:
                    var first = response(text: text)
                    first["done"] = false
                    var last = response(counts: includeUsage ? counts : nil)
                    last["message"] = ["role": "assistant"]
                    events = [first, last]
                }
                let json = try events.map { try ProviderUsageTests.json($0) }
                if self == .ollama { return json.joined(separator: "\n") + "\n" }
                return json.map { "data: \($0)\n\n" }.joined() + (self == .chat ? "data: [DONE]\n\n" : "")
            }
        }

        static func json(_ object: [String: Any]) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
        }

        @Test(arguments: Provider.allCases)
        func responseUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(counts: provider.counts)))
            let response = try await provider.makeSession().respond(to: "Hi")
            #expect(response.content == "Hello")
            #expect(response.usage == provider.expected)
        }

        @Test(arguments: Provider.allCases)
        func absentAndEmptyUsage(_ provider: Provider) async throws {
            for counts in [nil, [:]] as [[String: Any]?] {
                UsageURLProtocol.reset()
                UsageURLProtocol.enqueue(json: try Self.json(provider.response(counts: counts)))
                #expect(try await provider.makeSession().respond(to: "Hi").usage == .zero)
            }
        }

        @Test(arguments: Provider.allCases)
        func streamingUsageAndCollect(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.stream())
            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in provider.makeSession().streamResponse(to: "Hi") {
                snapshots.append(snapshot)
            }
            let finalContent: String? = snapshots.last?.content
            #expect(finalContent == "Hello")
            #expect(snapshots.last?.usage == provider.expected)
            #expect(snapshots.count >= 2)
            if provider == .chat {
                let body = try #require(UsageURLProtocol.recordedBodies.first)
                let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
                #expect((json?["stream_options"] as? [String: Bool])?["include_usage"] == true)
            }
            UsageURLProtocol.enqueue(json: try provider.stream())
            let session = provider.makeSession()
            let response = try await session.streamResponse(to: "Hi").collect()
            #expect(response.content == "Hello")
            #expect(response.usage == provider.expected)
            #expect(session.usage == provider.expected)
        }

        @Generable
        struct Answer { var answer: String }

        @Test(arguments: Provider.allCases)
        func structuredResponsesPreserveUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            let text = "{\"answer\":\"Hello\"}"
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: text, counts: provider.counts)))
            let response = try await provider.makeSession().respond(to: "Hi", generating: Answer.self)
            #expect(response.content.answer == "Hello")
            #expect(response.usage == provider.expected)
            UsageURLProtocol.enqueue(json: try provider.stream(text: text))
            let collected = try await provider.makeSession().streamResponse(to: "Hi", generating: Answer.self).collect()
            #expect(collected.content.answer == "Hello")
            #expect(collected.usage == provider.expected)
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func rawGeneratedContentPreservesUnquotedText(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.stream(text: "plain text"))
            let response = try await provider.makeSession().streamResponse(to: "Hi", generating: GeneratedContent.self)
                .collect()
            #expect(response.rawContent == GeneratedContent("plain text"))
            #expect(response.usage == provider.expected)
        }

        @Test(arguments: Provider.allCases)
        func partialUsageDefaultsUnknownCountsToZero(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            let key: String
            switch provider {
            case .chat: key = "completion_tokens"
            case .responses, .openResponses, .anthropic: key = "output_tokens"
            case .gemini: key = "candidatesTokenCount"
            case .ollama: key = "eval_count"
            }
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(counts: [key: 0])))
            let response = try await provider.makeSession().respond(to: "Hi")
            #expect(response.usage == .zero)
        }

        @Test(arguments: Provider.allCases)
        func emptyStreamingContentStillReportsUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.stream(text: ""))
            let response = try await provider.makeSession().streamResponse(to: "Hi").collect()
            #expect(response.content.isEmpty)
            #expect(response.usage == provider.expected)
        }

        @Test(arguments: Provider.allCases)
        func streamingWithoutUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.stream(includeUsage: false))
            let response = try await provider.makeSession().streamResponse(to: "Hi").collect()
            #expect(response.content == "Hello")
            #expect(response.usage == .zero)
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func toolRoundsAccumulateUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(counts: provider.counts, tool: true)))
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(counts: provider.counts)))
            let session = provider.makeSession(tools: [WeatherTool()])
            let response = try await session.respond(to: "Weather?")
            #expect(session.usage == response.usage)
            #expect(response.content == "Hello")
            #expect(response.usage.input.totalTokenCount == 200)
            #expect(response.usage.output.totalTokenCount == 40)
            #expect(response.usage.output.reasoningTokenCount == 10)
            #expect(response.usage.input.cachedTokenCount == (provider == .gemini ? 0 : 50))
            #expect(response.transcriptEntries.count == 2)
            #expect(UsageURLProtocol.recordedBodies.count == 2)
        }

        private struct StopDelegate: ToolExecutionDelegate {
            func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
                -> ToolExecutionDecision
            { .stop }
        }

        @Test(arguments: Provider.allCases)
        func stoppedToolCallsPreserveUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(counts: provider.counts, tool: true)))
            let session = provider.makeSession(tools: [WeatherTool()])
            session.toolExecutionDelegate = StopDelegate()
            let response = try await session.respond(to: "Weather?")
            #expect(response.content.isEmpty)
            #expect(response.usage == provider.expected)
            #expect(UsageURLProtocol.recordedBodies.count == 1)
        }
    }
    /// A `URLProtocol` that answers requests from a queue of canned responses
    /// and records every request body it sees,
    /// so request/response round trips can be asserted offline.
    private final class UsageURLProtocol: URLProtocol {
        struct Exchange: Sendable {
            var statusCode: Int = 200
            var body: Data
        }

        private struct State: Sendable {
            var pending: [Exchange] = []
            var recordedBodies: [Data] = []
        }

        private static let state = Locked(State())

        /// Discards queued responses and recorded bodies.
        static func reset() {
            state.withLock { $0 = State() }
        }

        /// Queues one JSON response,
        /// returned to the next request that arrives.
        static func enqueue(json: String, statusCode: Int = 200) {
            state.withLock { $0.pending.append(Exchange(statusCode: statusCode, body: Data(json.utf8))) }
        }

        /// The bodies of the requests seen so far,
        /// in order.
        static var recordedBodies: [Data] {
            state.withLock { $0.recordedBodies }
        }

        /// A session that routes every request to this protocol.
        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [UsageURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            // URLSession moves `httpBody` to `httpBodyStream`
            // before the protocol sees the request.
            let body = request.httpBody ?? request.httpBodyStream.map(Self.readAll) ?? Data()

            let exchange = Self.state.withLock { state -> Exchange? in
                state.recordedBodies.append(body)
                return state.pending.isEmpty ? nil : state.pending.removeFirst()
            }

            guard let exchange, let url = request.url else {
                client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
                return
            }

            let response = HTTPURLResponse(
                url: url,
                statusCode: exchange.statusCode,
                httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": request.value(forHTTPHeaderField: "Accept") == "text/event-stream"
                        ? "text/event-stream" : "application/json"
                ]
            )!

            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: exchange.body)
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}

        private static func readAll(_ stream: InputStream) -> Data {
            stream.open()
            defer { stream.close() }

            var data = Data()
            let bufferSize = 4096
            var buffer = [UInt8](repeating: 0, count: bufferSize)
            while true {
                let read = stream.read(&buffer, maxLength: bufferSize)
                if read <= 0 { break }
                data.append(buffer, count: read)
            }
            return data
        }
    }
#endif
