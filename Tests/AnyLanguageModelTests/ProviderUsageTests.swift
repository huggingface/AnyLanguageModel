import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Provider token usage", .serialized)
    struct ProviderUsageTests {
        enum Provider: CaseIterable, Equatable, Sendable {
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
                    return [
                        "promptTokenCount": 100, "cachedContentTokenCount": 25,
                        "candidatesTokenCount": 20, "thoughtsTokenCount": 5,
                        "toolUsePromptTokenCount": 10, "totalTokenCount": 135,
                    ]
                case .ollama:
                    return ["prompt_eval_count": 100, "eval_count": 20]
                }
            }

            var expected: LanguageModelSession.Usage {
                .init(
                    input: .init(
                        totalTokenCount: self == .anthropic ? 135 : (self == .gemini ? 110 : 100),
                        cachedTokenCount: self == .ollama ? 0 : 25
                    ),
                    output: .init(
                        totalTokenCount: self == .gemini ? 25 : 20,
                        reasoningTokenCount: self == .anthropic || self == .ollama ? 0 : 5
                    ),
                    metadata: self == .anthropic ? ["cache_creation_input_tokens": 10] : [:]
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

            func toolStream(includeUsage: Bool = true, city: String = "Paris", callID: String = "call_1") throws
                -> String
            {
                let toolResponse = response(text: "", counts: includeUsage ? counts : nil, tool: true)
                let events: [[String: Any]]
                switch self {
                case .chat:
                    events =
                        [
                            [
                                "id": "test",
                                "choices": [
                                    [
                                        "delta": [
                                            "tool_calls": [
                                                [
                                                    "index": 0, "id": "call_1", "type": "function",
                                                    "function": ["name": "getWeather", "arguments": "{\"city\":"],
                                                ]
                                            ]
                                        ]
                                    ]
                                ],
                            ],
                            [
                                "id": "test",
                                "choices": [
                                    [
                                        "delta": [
                                            "tool_calls": [
                                                [
                                                    "index": 0, "function": ["arguments": "\"Paris\"}"],
                                                ]
                                            ]
                                        ], "finish_reason": "tool_calls",
                                    ]
                                ],
                            ],
                        ] + (includeUsage ? [["id": "test", "choices": [], "usage": counts]] : [])
                case .responses, .openResponses:
                    events = [["type": "response.completed", "response": toolResponse]]
                case .anthropic:
                    events = [
                        ["type": "message_start", "message": response(text: "", counts: includeUsage ? counts : nil)],
                        [
                            "type": "content_block_start", "index": 0,
                            "content_block": [
                                "type": "tool_use", "id": "call_1", "name": "getWeather", "input": [:],
                            ],
                        ],
                        [
                            "type": "content_block_delta", "index": 0,
                            "delta": [
                                "type": "input_json_delta", "partial_json": "{\"city\":",
                            ],
                        ],
                        [
                            "type": "content_block_delta", "index": 0,
                            "delta": [
                                "type": "input_json_delta", "partial_json": "\"Paris\"}",
                            ],
                        ],
                        ["type": "content_block_stop", "index": 0],
                        ["type": "message_stop"],
                    ]
                case .gemini, .ollama:
                    events = [toolResponse]
                }
                return try events.map { event in
                    let json = try ProviderUsageTests.json(event)
                        .replacingOccurrences(of: "Paris", with: city)
                        .replacingOccurrences(of: "call_1", with: callID)
                    return self == .ollama ? json + "\n" : "data: \(json)\n\n"
                }.joined() + (self == .chat ? "data: [DONE]\n\n" : "")
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
                    early["totalTokenCount"] = 122
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

        @Test(arguments: Provider.allCases, [false, true])
        func streamingToolRound(_ provider: Provider, _ collect: Bool) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream(includeUsage: false))
            UsageURLProtocol.enqueue(json: try provider.stream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            let stream = session.streamResponse(to: "Weather?")
            if collect {
                let response = try await stream.collect()
                #expect(response.content == "Hello")
                #expect(response.transcriptEntries.count == 2)
                #expect(response.usage == provider.expected)
            } else {
                var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
                for try await snapshot in stream { snapshots.append(snapshot) }
                let finalContent: String? = snapshots.last?.content
                #expect(finalContent == "Hello")
                #expect(snapshots.last?.transcriptEntries.count == 2)
                #expect(snapshots.last?.usage == provider.expected)
            }
            #expect(UsageURLProtocol.recordedBodies.count == 2)
            #expect(session.transcript.count == 4)
            #expect(tool.cities.withLock { $0 } == ["Paris"])
            let body = try #require(UsageURLProtocol.recordedBodies.last)
            #expect(String(decoding: body, as: UTF8.self).contains("The weather in Paris is sunny"))
            #expect(session.usage == provider.expected)
            #expect(!session.isResponding)
        }

        private struct RecordingWeatherTool: Tool {
            let name = "getWeather"
            let description = "Get the weather"
            let cities = Locked<[String]>([])
            var fails = false

            func call(arguments: WeatherTool.Arguments) async throws -> String {
                cities.withLock { $0.append(arguments.city) }
                if fails { throw ToolFailure.failed }
                return try await WeatherTool().call(arguments: arguments)
            }
        }

        private enum ToolFailure: Error { case failed }

        private struct OutputDelegate: ToolExecutionDelegate {
            func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
                -> ToolExecutionDecision
            { .provideOutput([.text(.init(content: "Cached weather"))]) }
        }

        @Test(arguments: Provider.allCases)
        func streamedToolRoundsAccumulateUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            UsageURLProtocol.enqueue(json: try provider.toolStream(city: "London", callID: "call_2"))
            UsageURLProtocol.enqueue(json: try provider.stream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in session.streamResponse(to: "Weather?") { snapshots.append(snapshot) }
            let last = try #require(snapshots.last)
            #expect(last.content == "Hello")
            var expected = provider.expected
            expected.add(provider.expected)
            expected.add(provider.expected)
            #expect(last.usage == expected)
            #expect(session.usage == expected)
            #expect(last.transcriptEntries.count == 4)
            #expect(session.transcript.count == 6)
            #expect(tool.cities.withLock { $0 } == ["Paris", "London"])
            #expect(UsageURLProtocol.recordedBodies.count == 3)
            for (previous, next) in zip(snapshots, snapshots.dropFirst()) {
                #expect(next.usage.totalTokenCount >= previous.usage.totalTokenCount)
                #expect(next.transcriptEntries.count >= previous.transcriptEntries.count)
            }
        }

        @Test(arguments: Provider.allCases, [false, true])
        func streamedToolsThenStructuredContent(_ provider: Provider, _ dynamicSchema: Bool) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            UsageURLProtocol.enqueue(json: try provider.stream(text: "{\"answer\":\"Hello\"}"))
            let session = provider.makeSession(tools: [WeatherTool()])
            if dynamicSchema {
                let response = try await session.streamResponse(to: "Weather?", schema: Answer.generationSchema)
                    .collect()
                #expect(try response.content.value(String.self, forProperty: "answer") == "Hello")
                #expect(response.transcriptEntries.count == 2)
            } else {
                let response = try await session.streamResponse(to: "Weather?", generating: Answer.self).collect()
                #expect(response.content.answer == "Hello")
                #expect(response.transcriptEntries.count == 2)
            }
            var expected = provider.expected
            expected.add(provider.expected)
            #expect(session.usage == expected)
            #expect(UsageURLProtocol.recordedBodies.count == 2)
        }

        @Test(arguments: Provider.allCases)
        func stoppedStreamedTools(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            session.toolExecutionDelegate = StopDelegate()
            let response = try await session.streamResponse(to: "Weather?").collect()
            #expect(response.content.isEmpty)
            #expect(response.transcriptEntries.count == 1)
            #expect(response.usage == provider.expected)
            #expect(tool.cities.withLock { $0.isEmpty })
            #expect(UsageURLProtocol.recordedBodies.count == 1)
            #expect(!session.isResponding)
        }

        @Test func openResponsesToolHistoryUsesTopLevelFunctionCallItems() async throws {
            let provider = Provider.openResponses
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            UsageURLProtocol.enqueue(json: try provider.stream())
            UsageURLProtocol.enqueue(json: try provider.stream())
            let session = provider.makeSession(tools: [WeatherTool()])
            _ = try await session.streamResponse(to: "Weather?").collect()
            _ = try await session.streamResponse(to: "And tomorrow?").collect()
            #expect(UsageURLProtocol.recordedBodies.count == 3)

            let body = try #require(UsageURLProtocol.recordedBodies.last)
            let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
            let input = try #require(json["input"] as? [[String: Any]])
            let types = input.map { $0["type"] as? String }
            let callIndex = try #require(types.firstIndex(of: "function_call"))
            let outputIndex = try #require(types.firstIndex(of: "function_call_output"))
            #expect(types.filter { $0 == "function_call" }.count == 1)
            #expect(callIndex < outputIndex)
            let call = input[callIndex]
            #expect(call["call_id"] as? String == "call_1")
            #expect(call["name"] as? String == "getWeather")
            let arguments = try #require(call["arguments"] as? String)
            let decodedArguments = try JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: String]
            #expect(decodedArguments == ["city": "Paris"])
            #expect(call["id"] == nil)
            #expect(input[outputIndex]["call_id"] as? String == "call_1")
            for item in input where item["type"] as? String == "message" {
                let content = item["content"] as? [[String: Any]] ?? []
                #expect(!content.contains { $0["type"] as? String == "function_call" })
            }
        }

        @Test(arguments: Provider.allCases)
        func streamedToolOutputOverride(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            UsageURLProtocol.enqueue(json: try provider.stream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            session.toolExecutionDelegate = OutputDelegate()
            let response = try await session.streamResponse(to: "Weather?").collect()
            #expect(response.content == "Hello")
            #expect(tool.cities.withLock { $0.isEmpty })
            let body = try #require(UsageURLProtocol.recordedBodies.last)
            #expect(String(decoding: body, as: UTF8.self).contains("Cached weather"))
            #expect(UsageURLProtocol.recordedBodies.count == 2)
        }

        @Test(arguments: Provider.allCases)
        func streamedToolFailure(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            let tool = RecordingWeatherTool(fails: true)
            let session = provider.makeSession(tools: [tool])
            do {
                _ = try await session.streamResponse(to: "Weather?").collect()
                Issue.record("Expected the tool error")
            } catch let error as LanguageModelSession.ToolCallError {
                #expect(error.underlyingError is ToolFailure)
            }
            #expect(tool.cities.withLock { $0 } == ["Paris"])
            #expect(UsageURLProtocol.recordedBodies.count == 1)
            #expect(!session.isResponding)
        }

        @Test(arguments: [Provider.chat, .anthropic])
        func interleavedStreamedToolArguments(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            let first = try provider.toolStream(includeUsage: false).components(separatedBy: "\n\n")
                .filter { !$0.isEmpty && !$0.contains("[DONE]") }
            let second = try provider.toolStream(includeUsage: false, city: "London", callID: "call_2")
                .replacingOccurrences(of: "\"index\":0", with: "\"index\":1")
                .components(separatedBy: "\n\n")
                .filter { !$0.isEmpty && !$0.contains("[DONE]") }
            let events = zip(first, second).flatMap { [$0, $1] }.filter {
                !$0.contains("message_stop") && !$0.contains("message_start")
            }
            let end = provider == .chat ? "data: [DONE]\n\n" : "data: {\"type\":\"message_stop\"}\n\n"
            UsageURLProtocol.enqueue(json: events.joined(separator: "\n\n") + "\n\n" + end)
            UsageURLProtocol.enqueue(json: try provider.stream())
            let tool = RecordingWeatherTool()
            let response = try await provider.makeSession(tools: [tool]).streamResponse(to: "Weather?").collect()
            #expect(response.content == "Hello")
            #expect(tool.cities.withLock { $0 } == ["Paris", "London"])
            #expect(response.transcriptEntries.count == 3)
            let body = try #require(UsageURLProtocol.recordedBodies.last)
            let json = String(decoding: body, as: UTF8.self)
            #expect(json.contains("call_1"))
            #expect(json.contains("call_2"))
            #expect(json.contains("The weather in Paris is sunny"))
            #expect(json.contains("The weather in London is sunny"))
        }

        private struct CancellableWeatherTool: Tool {
            let name = "getWeather"
            let description = "Get the weather"
            let started: AsyncStream<Void>.Continuation
            let finished: AsyncStream<Void>.Continuation

            func call(arguments: WeatherTool.Arguments) async throws -> String {
                started.yield(())
                defer { finished.yield(()) }
                try await Task.sleep(for: .seconds(60))
                return "Unexpected result"
            }
        }

        @Test(arguments: Provider.allCases)
        func cancelledToolStreamDoesNotRequestNextRound(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            let (started, startContinuation) = AsyncStream<Void>.makeStream()
            let (finished, finishContinuation) = AsyncStream<Void>.makeStream()
            let session = provider.makeSession(tools: [
                CancellableWeatherTool(started: startContinuation, finished: finishContinuation)
            ])
            let consumer = Task { try await session.streamResponse(to: "Weather?").collect() }
            for await _ in started { break }
            consumer.cancel()
            _ = await consumer.result
            for await _ in finished { break }
            #expect(UsageURLProtocol.recordedBodies.count == 1)
        }

        @Test func customChatEndpointOmitsStreamOptions() async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Provider.chat.stream(includeUsage: false))
            let model = OpenAILanguageModel(
                baseURL: URL(string: "https://example.com/v1")!,
                apiKey: "test",
                model: "test",
                session: UsageURLProtocol.makeSession()
            )
            let response = try await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()
            #expect(response.content == "Hello")
            #expect(response.usage == .zero)
            let body = try #require(UsageURLProtocol.recordedBodies.first)
            let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
            #expect(json["stream_options"] == nil)
        }

        @Test(arguments: ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"])
        func partialAnthropicInputUsage(_ key: String) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(Provider.anthropic.response(counts: [key: 10])))
            let response = try await Provider.anthropic.makeSession().respond(to: "Hi")
            #expect(response.usage.input.totalTokenCount == 10)
            #expect(response.usage.input.cachedTokenCount == (key == "cache_read_input_tokens" ? 10 : 0))
            #expect(response.usage.output.totalTokenCount == 0)
            #expect(
                response.usage.metadata["cache_creation_input_tokens"]
                    == (key == "cache_creation_input_tokens" ? GeneratedContent(10) : nil)
            )
        }

        @Test(
            arguments: ["promptTokenCount", "toolUsePromptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"],
            [0, 10]
        )
        func partialGeminiUsage(_ key: String, _ count: Int) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(Provider.gemini.response(counts: [key: count])))
            let response = try await Provider.gemini.makeSession().respond(to: "Hi")
            let isInput = key == "promptTokenCount" || key == "toolUsePromptTokenCount"
            #expect(response.usage.input.totalTokenCount == (isInput ? count : 0))
            #expect(response.usage.output.totalTokenCount == (isInput ? 0 : count))
            #expect(response.usage.output.reasoningTokenCount == (key == "thoughtsTokenCount" ? count : 0))
            #expect(response.usage.totalTokenCount == count)
        }

        @Test func geminiStreamingRetainsOmittedUsageTotals() async throws {
            UsageURLProtocol.reset()
            let events: [[String: Any]] = [
                Provider.gemini.response(counts: ["promptTokenCount": 100, "toolUsePromptTokenCount": 10]),
                ["usageMetadata": ["candidatesTokenCount": 20, "thoughtsTokenCount": 5]],
                ["usageMetadata": ["cachedContentTokenCount": 25]],
            ]
            let stream = try events.map { "data: \(try Self.json($0))\n\n" }.joined()
            UsageURLProtocol.enqueue(json: stream)
            let session = Provider.gemini.makeSession()
            let response = try await session.streamResponse(to: "Hi").collect()
            #expect(response.content == "Hello")
            #expect(response.usage == Provider.gemini.expected)
            #expect(response.usage.totalTokenCount == 135)
            #expect(session.usage == response.usage)
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
            #expect(response.usage.input.totalTokenCount == (provider == .gemini ? 220 : 200))
            #expect(response.usage.output.totalTokenCount == (provider == .gemini ? 50 : 40))
            #expect(response.usage.output.reasoningTokenCount == 10)
            #expect(response.usage.input.cachedTokenCount == 50)
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
