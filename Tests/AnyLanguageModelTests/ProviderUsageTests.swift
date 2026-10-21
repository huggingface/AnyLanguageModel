import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Provider token usage", .serialized)
    struct ProviderUsageTests {
        enum Provider: CaseIterable, Equatable, Sendable {
            case chat, responses, openResponses, anthropic, gemini, ollama

            func makeModel() -> any LanguageModel {
                let http = UsageURLProtocol.makeSession()
                switch self {
                case .chat, .responses:
                    return OpenAILanguageModel(
                        apiKey: "test",
                        model: "test",
                        apiVariant: self == .chat ? .chatCompletions : .responses,
                        session: http
                    )
                case .openResponses:
                    return OpenResponsesLanguageModel(
                        baseURL: URL(string: "https://example.com/v1")!,
                        apiKey: "test",
                        model: "test",
                        session: http
                    )
                case .anthropic:
                    return AnthropicLanguageModel(apiKey: "test", model: "test", session: http)
                case .gemini:
                    return GeminiLanguageModel(apiKey: "test", model: "test", session: http)
                case .ollama:
                    return OllamaLanguageModel(model: "test", session: http)
                }
            }

            func makeSession(tools: [any Tool] = [], transcript: Transcript = .init()) -> LanguageModelSession {
                LanguageModelSession(model: makeModel(), tools: tools, transcript: transcript)
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
                    let message: [String: Any] = [
                        "type": "message", "content": [["type": "output_text", "text": text]],
                    ]
                    let output: [[String: Any]] =
                        tool
                        ? (text.isEmpty ? [] : [message]) + [
                            [
                                "type": "function_call", "call_id": "call_1", "name": "getWeather",
                                "arguments": "{\"city\":\"Paris\"}",
                            ]
                        ]
                        : [message]
                    result = ["id": "test", "output": output]
                case .anthropic:
                    let textBlock: [String: Any] = ["type": "text", "text": text]
                    let content: [[String: Any]] =
                        tool
                        ? (text.isEmpty ? [] : [textBlock]) + [
                            ["type": "tool_use", "id": "call_1", "name": "getWeather", "input": ["city": "Paris"]]
                        ]
                        : [textBlock]
                    result = [
                        "id": "test", "type": "message", "role": "assistant", "model": "test", "content": content,
                    ]
                case .gemini:
                    let parts: [[String: Any]] =
                        tool
                        ? (text.isEmpty ? [] : [["text": text]]) + [
                            ["functionCall": ["name": "getWeather", "args": ["city": "Paris"]]]
                        ]
                        : [["text": text]]
                    result = ["candidates": [["content": ["role": "model", "parts": parts]]]]
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

            /// A streamed tool round, optionally with `text` before the tool call.
            func toolStream(
                text: String = "",
                includeUsage: Bool = true,
                city: String = "Paris",
                callID: String = "call_1"
            ) throws -> String {
                let toolResponse = response(text: text, counts: includeUsage ? counts : nil, tool: true)
                var events: [[String: Any]]
                switch self {
                case .chat:
                    events =
                        (text.isEmpty ? [] : [["id": "test", "choices": [["delta": ["content": text]]]]]) + [
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
                    events =
                        (text.isEmpty ? [] : [["type": "response.output_text.delta", "delta": text]])
                        + [["type": "response.completed", "response": toolResponse]]
                case .anthropic:
                    let toolIndex = text.isEmpty ? 0 : 1
                    events = [
                        ["type": "message_start", "message": response(text: "", counts: includeUsage ? counts : nil)]
                    ]
                    if !text.isEmpty {
                        events += [
                            [
                                "type": "content_block_start", "index": 0,
                                "content_block": ["type": "text", "text": ""],
                            ],
                            [
                                "type": "content_block_delta", "index": 0,
                                "delta": ["type": "text_delta", "text": text],
                            ],
                            ["type": "content_block_stop", "index": 0],
                        ]
                    }
                    events += [
                        [
                            "type": "content_block_start", "index": toolIndex,
                            "content_block": [
                                "type": "tool_use", "id": "call_1", "name": "getWeather", "input": [:],
                            ],
                        ],
                        [
                            "type": "content_block_delta", "index": toolIndex,
                            "delta": [
                                "type": "input_json_delta", "partial_json": "{\"city\":",
                            ],
                        ],
                        [
                            "type": "content_block_delta", "index": toolIndex,
                            "delta": [
                                "type": "input_json_delta", "partial_json": "\"Paris\"}",
                            ],
                        ],
                        ["type": "content_block_stop", "index": toolIndex],
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

        @Test(arguments: Provider.allCases)
        func streamedToolRoundsKeepEarlierText(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream(text: "Checking. "))
            UsageURLProtocol.enqueue(json: try provider.stream(text: "Sunny"))
            let session = provider.makeSession(tools: [RecordingWeatherTool()])
            var contents: [String] = []
            for try await snapshot in session.streamResponse(to: "Weather?") { contents.append(snapshot.content) }
            #expect(contents.last == "Checking. Sunny")
            for (previous, next) in zip(contents, contents.dropFirst()) {
                #expect(next.hasPrefix(previous))
            }
            #expect(UsageURLProtocol.recordedBodies.count == 2)
            let followUp = try #require(UsageURLProtocol.recordedBodies.last)
            #expect(String(decoding: followUp, as: UTF8.self).contains("Checking. "))
        }

        @Test(arguments: Provider.allCases)
        func streamedStructuredContentOmitsEarlierText(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream(text: "Checking. "))
            UsageURLProtocol.enqueue(json: try provider.stream(text: "{\"answer\":\"Hello\"}"))
            let session = provider.makeSession(tools: [WeatherTool()])
            let response = try await session.streamResponse(to: "Weather?", generating: Answer.self).collect()
            #expect(response.content.answer == "Hello")
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func toolRoundsKeepEarlierText(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "Checking. ", tool: true)))
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "Sunny")))
            let session = provider.makeSession(tools: [WeatherTool()])
            let response = try await session.respond(to: "Weather?")
            #expect(response.content == "Checking. Sunny")
            #expect(UsageURLProtocol.recordedBodies.count == 2)
            let followUp = try #require(UsageURLProtocol.recordedBodies.last)
            #expect(String(decoding: followUp, as: UTF8.self).contains("Checking. "))
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

        @Test(arguments: [Provider.anthropic, .gemini], [false, true])
        func localToolImagesSurviveDeletionAndTranscriptRestoration(_ provider: Provider, _ streaming: Bool)
            async throws
        {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            if streaming || provider == .gemini {
                UsageURLProtocol.enqueue(json: try streaming ? provider.stream() : Self.json(provider.response()))
            }
            UsageURLProtocol.enqueue(json: try Self.json(provider.response()))
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("png")
            try testImageData.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let session = provider.makeSession(tools: [LocalImageWeatherTool(url: url)])
            session.toolExecutionDelegate = DeleteImageAfterToolOutput(url: url)

            if streaming {
                _ = try await session.streamResponse(to: "Weather?").collect()
            } else {
                _ = try await session.respond(to: "Weather?")
            }
            #expect(!FileManager.default.fileExists(atPath: url.path))
            let archived = try JSONEncoder().encode(session.transcript)
            let restored = provider.makeSession(transcript: try JSONDecoder().decode(Transcript.self, from: archived))
            _ = try await restored.respond(to: "Review the image")

            #expect(UsageURLProtocol.recordedBodies.count == (streaming || provider == .gemini ? 3 : 2))
            for body in UsageURLProtocol.recordedBodies.dropFirst() {
                let request = try JSONDecoder().decode([String: JSONValue].self, from: body)
                if provider == .gemini {
                    let contents = try #require(request["contents"]?.arrayValue)
                    let results = contents.flatMap { $0.objectValue?["parts"]?.arrayValue ?? [] }
                        .compactMap { $0.objectValue?["functionResponse"]?.objectValue }
                    let parts = try #require(results.first?["parts"]?.arrayValue)
                    let inlineData = try #require(parts.first?.objectValue?["inlineData"]?.objectValue)
                    #expect(inlineData["mime_type"] == .string("image/png"))
                    #expect(inlineData["data"] == .string(testImageData.base64EncodedString()))
                    continue
                }
                let messages = try #require(request["messages"]?.arrayValue)
                let results = messages.flatMap { $0.objectValue?["content"]?.arrayValue ?? [] }
                    .filter { $0.objectValue?["type"] == .string("tool_result") }
                let content = try #require(results.first?.objectValue?["content"]?.arrayValue)
                let image = try #require(content.first { $0.objectValue?["type"] == .string("image") })
                let source = try #require(image.objectValue?["source"]?.objectValue)
                #expect(source["type"] == .string("base64"))
                #expect(source["media_type"] == .string("image/png"))
                #expect(source["data"] == .string(testImageData.base64EncodedString()))
                #expect(source["url"] == nil)
            }
        }

        @Test(arguments: [Provider.anthropic, .gemini], [false, true])
        func toolOutputImageMIMEsPreserveOnlySupportedWireParts(_ provider: Provider, _ streaming: Bool) async throws {
            for mimeType in [
                "image/png", "image/jpeg", "image/gif", "image/webp", "image/tiff", "image/svg+xml", "image/avif",
                "application/octet-stream",
            ] {
                let supported =
                    ["image/png", "image/jpeg", "image/webp"].contains(mimeType)
                    || (provider == .anthropic && mimeType == "image/gif")
                let segments: [Transcript.Segment] = [
                    .text(.init(content: "before")),
                    .image(.init(data: testImageData, mimeType: mimeType)),
                    .text(.init(content: "after")),
                ]
                let session = provider.makeSession(tools: [WeatherTool()])
                session.toolExecutionDelegate = ImageSegmentsDelegate(segments: segments)
                try await checkImageToolContinuationAndReplay(
                    provider,
                    streaming: streaming,
                    session: session,
                    expectedTexts: supported ? ["before", "after"] : ["before", "<image>", "after"],
                    expectedMIME: supported ? mimeType : nil
                )
                let output = try #require(
                    session.transcript.compactMap { entry -> Transcript.ToolOutput? in
                        if case .toolOutput(let output) = entry { return output }
                        return nil
                    }.first
                )
                #expect(output.segments == segments)
            }
        }

        @Test(arguments: [Provider.anthropic, .gemini], [false, true])
        func localToolImageFormatsSnapshotBeforePlaceholder(_ provider: Provider, _ streaming: Bool) async throws {
            for (fileExtension, mimeType) in [("tiff", "image/tiff"), ("svg", "image/svg+xml"), ("gif", "image/gif")] {
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString).appendingPathExtension(fileExtension)
                try testImageData.write(to: url)
                defer { try? FileManager.default.removeItem(at: url) }
                let session = provider.makeSession(tools: [LocalImageWeatherTool(url: url)])
                session.toolExecutionDelegate = DeleteImageAfterToolOutput(url: url, mimeType: mimeType)
                let supported = provider == .anthropic && fileExtension == "gif"
                try await checkImageToolContinuationAndReplay(
                    provider,
                    streaming: streaming,
                    session: session,
                    expectedTexts: supported ? [] : ["<image>"],
                    expectedMIME: supported ? mimeType : nil
                )
                #expect(!FileManager.default.fileExists(atPath: url.path))
                let output = try #require(
                    session.transcript.compactMap { entry -> Transcript.ToolOutput? in
                        if case .toolOutput(let output) = entry { return output }
                        return nil
                    }.first
                )
                guard case .image(let image)? = output.segments.first else {
                    Issue.record("Wire placeholder must not replace the canonical image")
                    continue
                }
                #expect(image.source == .data(testImageData, mimeType: mimeType))
            }
        }

        private func checkImageToolContinuationAndReplay(
            _ provider: Provider,
            streaming: Bool,
            session: LanguageModelSession,
            expectedTexts: [String],
            expectedMIME: String?
        ) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            if streaming || provider == .gemini {
                UsageURLProtocol.enqueue(json: try streaming ? provider.stream() : Self.json(provider.response()))
            }
            if streaming {
                _ = try await session.streamResponse(to: "Weather?").collect()
            } else {
                _ = try await session.respond(to: "Weather?")
            }
            let archived = try JSONEncoder().encode(session.transcript)
            let transcript = try JSONDecoder().decode(Transcript.self, from: archived)
            #expect(transcript == session.transcript)
            let restored = provider.makeSession(transcript: transcript)
            UsageURLProtocol.enqueue(json: try streaming ? provider.stream() : Self.json(provider.response()))
            if streaming {
                _ = try await restored.streamResponse(to: "Review the image").collect()
            } else {
                _ = try await restored.respond(to: "Review the image")
            }
            #expect(UsageURLProtocol.recordedBodies.count == (streaming || provider == .gemini ? 3 : 2))
            for body in UsageURLProtocol.recordedBodies.dropFirst() {
                try expectToolImage(in: body, provider: provider, texts: expectedTexts, mimeType: expectedMIME)
            }
        }

        private func expectToolImage(in body: Data, provider: Provider, texts: [String], mimeType: String?) throws {
            let request = try JSONDecoder().decode([String: JSONValue].self, from: body)
            if provider == .gemini {
                let contents = try #require(request["contents"]?.arrayValue)
                let results = contents.flatMap { $0.objectValue?["parts"]?.arrayValue ?? [] }
                    .compactMap { $0.objectValue?["functionResponse"]?.objectValue }
                try #require(results.count == 1)
                let result = try #require(results.first)
                #expect(
                    result["response"]?.objectValue?["result"]
                        == (texts.isEmpty ? nil : .string(texts.joined(separator: "\n")))
                )
                let parts = result["parts"]?.arrayValue ?? []
                #expect(parts.count == (mimeType == nil ? 0 : 1))
                if let mimeType {
                    let image = try #require(parts.first?.objectValue?["inlineData"]?.objectValue)
                    #expect(image["mime_type"] == .string(mimeType))
                    #expect(image["data"] == .string(testImageData.base64EncodedString()))
                }
            } else {
                let messages = try #require(request["messages"]?.arrayValue)
                let results = messages.flatMap { $0.objectValue?["content"]?.arrayValue ?? [] }
                    .filter { $0.objectValue?["type"] == .string("tool_result") }
                try #require(results.count == 1)
                let content = try #require(results.first?.objectValue?["content"]?.arrayValue)
                #expect(content.compactMap { $0.objectValue?["text"]?.stringValue } == texts)
                let images = content.filter { $0.objectValue?["type"] == .string("image") }
                #expect(images.count == (mimeType == nil ? 0 : 1))
                if let mimeType {
                    let source = try #require(images.first?.objectValue?["source"]?.objectValue)
                    #expect(source["media_type"] == .string(mimeType))
                    #expect(source["data"] == .string(testImageData.base64EncodedString()))
                    #expect(source["type"] == .string("base64"))
                    #expect(
                        content.map { $0.objectValue?["type"]?.stringValue }
                            == (texts.isEmpty ? ["image"] : ["text", "image", "text"])
                    )
                } else {
                    #expect(content.allSatisfy { $0.objectValue?["type"] == .string("text") })
                }
            }
        }

        private struct ImageSegmentsDelegate: ToolExecutionDelegate {
            let segments: [Transcript.Segment]

            func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
                -> ToolExecutionDecision
            { .provideOutput(segments) }
        }

        @Test(arguments: [Provider.anthropic, .gemini])
        func legacyUnsupportedToolImageURLsUsePlaceholder(_ provider: Provider) async throws {
            for fileExtension in ["tiff", "svg"] {
                UsageURLProtocol.reset()
                UsageURLProtocol.enqueue(json: try Self.json(provider.response()))
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString).appendingPathExtension(fileExtension)
                let transcript = Transcript(entries: [
                    .toolCalls(
                        .init([.init(id: "call_1", toolName: "getWeather", arguments: GeneratedContent("Paris"))])
                    ),
                    .toolOutput(
                        .init(id: "call_1", toolName: "getWeather", segments: [.image(.init(source: .url(url)))])
                    ),
                ])
                let archived = try JSONEncoder().encode(transcript)
                let decoded = try JSONDecoder().decode(Transcript.self, from: archived)
                #expect(decoded == transcript)
                _ = try await provider.makeSession(transcript: decoded).respond(to: "Review the image")
                try #require(UsageURLProtocol.recordedBodies.count == 1)
                try expectToolImage(
                    in: try #require(UsageURLProtocol.recordedBodies.first),
                    provider: provider,
                    texts: ["<image>"],
                    mimeType: nil
                )
            }
        }

        @Test(arguments: [Provider.anthropic, .gemini], [false, true])
        func missingToolImageFilesStillFail(_ provider: Provider, _ streaming: Bool) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
            let session = provider.makeSession(tools: [LocalImageWeatherTool(url: url)])
            do {
                if streaming {
                    _ = try await session.streamResponse(to: "Weather?").collect()
                } else {
                    _ = try await session.respond(to: "Weather?")
                }
                Issue.record("A read failure must not become an image placeholder")
            } catch let error as LanguageModelSession.ToolCallError {
                #expect((error.underlyingError as NSError).domain == NSCocoaErrorDomain)
                #expect((error.underlyingError as NSError).code == NSFileReadNoSuchFileError)
            }
            #expect(UsageURLProtocol.recordedBodies.count == 1)
        }

        @Test(arguments: [Provider.anthropic, .gemini])
        func legacySupportedToolImageReadFailureIsNotAPlaceholder(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
            let transcript = Transcript(entries: [
                .toolCalls(.init([.init(id: "call_1", toolName: "getWeather", arguments: GeneratedContent("Paris"))])),
                .toolOutput(.init(id: "call_1", toolName: "getWeather", segments: [.image(.init(source: .url(url)))])),
            ])
            do {
                _ = try await provider.makeSession(transcript: transcript).respond(to: "Review the image")
                Issue.record("A supported image read failure must remain explicit")
            } catch {
                #expect((error as NSError).domain == NSCocoaErrorDomain)
                #expect((error as NSError).code == NSFileReadNoSuchFileError)
            }
            #expect(UsageURLProtocol.recordedBodies.isEmpty)
        }

        @Test(arguments: [Provider.anthropic, .gemini])
        func unsupportedUserImageMIMEsRemainImages(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(provider.response()))
            let transcript = Transcript(entries: [
                .prompt(.init(segments: [.image(.init(data: testImageData, mimeType: "image/tiff"))]))
            ])
            _ = try await provider.makeSession(transcript: transcript).respond(to: "Review the image")
            let body = try #require(UsageURLProtocol.recordedBodies.first)
            let request = try JSONDecoder().decode([String: JSONValue].self, from: body)
            if provider == .gemini {
                let images =
                    request["contents"]?.arrayValue?.flatMap { $0.objectValue?["parts"]?.arrayValue ?? [] }
                    .compactMap { $0.objectValue?["inlineData"]?.objectValue } ?? []
                #expect(images.first?["mime_type"] == .string("image/tiff"))
            } else {
                let images =
                    request["messages"]?.arrayValue?.flatMap { $0.objectValue?["content"]?.arrayValue ?? [] }
                    .filter { $0.objectValue?["type"] == .string("image") } ?? []
                #expect(images.first?.objectValue?["source"]?.objectValue?["media_type"] == .string("image/tiff"))
            }
        }

        private struct DeleteImageAfterToolOutput: ToolExecutionDelegate {
            let url: URL
            var mimeType = "image/png"

            func didExecuteToolCall(
                _ toolCall: Transcript.ToolCall,
                output: Transcript.ToolOutput,
                in session: LanguageModelSession
            ) async {
                guard case .image(let image)? = output.segments.first else {
                    Issue.record("Missing image in Tool output")
                    return
                }
                #expect(image.source == .data(testImageData, mimeType: mimeType))
                do { try FileManager.default.removeItem(at: url) } catch { Issue.record(error) }
            }
        }

        private struct LocalImageWeatherTool: Tool {
            let name = "getWeather"
            let description = "Get weather as an image"
            let url: URL

            func call(arguments: WeatherTool.Arguments) async throws -> Prompt {
                Prompt { Attachment(imageURL: url) }
            }
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

        @Test(arguments: Provider.allCases)
        func stoppedStreamedToolsWithoutEmptyObjectContent(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            session.toolExecutionDelegate = StopDelegate()
            var snapshots: [LanguageModelSession.ResponseStream<Int>.Snapshot] = []
            for try await snapshot in session.streamResponse(to: "Weather?", generating: Int.self) {
                snapshots.append(snapshot)
            }
            let last = try #require(snapshots.last)
            let content: Int = last.content
            #expect(content == 0)
            #expect(last.transcriptEntries.count == 1)
            #expect(last.usage == provider.expected)
            #expect(session.transcript.count == 3)

            UsageURLProtocol.enqueue(json: try provider.toolStream())
            let arraySession = provider.makeSession(tools: [tool])
            arraySession.toolExecutionDelegate = StopDelegate()
            let response = try await arraySession.streamResponse(to: "Weather?", generating: [String].self).collect()
            #expect(response.content.isEmpty)
            #expect(response.transcriptEntries.count == 1)
            #expect(tool.cities.withLock { $0.isEmpty })
            #expect(UsageURLProtocol.recordedBodies.count == 2)
        }

        @Test(arguments: Provider.allCases)
        func stoppedStreamedToolsWithoutDecodableEmptyContentThrow(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            let session = provider.makeSession(tools: [RecordingWeatherTool()])
            session.toolExecutionDelegate = StopDelegate()
            var snapshotCount = 0
            var thrownError: (any Error)?
            do {
                for try await _ in session.streamResponse(to: "Weather?", generating: Forecast.self) {
                    snapshotCount += 1
                }
            } catch {
                thrownError = error
            }
            #expect(snapshotCount == 0)
            #expect(thrownError as? GeneratedContentError == .typeMismatch)
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

        @Generable
        enum Sky { case clear, cloudy }

        @Generable
        struct Forecast { var sky: Sky }

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

        /// Returns what `body` throws when it goes past its tool-round limit, if it does.
        static func toolRoundLimitExceeded(
            _ body: () async throws -> Void
        ) async -> LanguageModelSession.ToolRoundLimitExceeded? {
            do {
                try await body()
                return nil
            } catch let error as LanguageModelSession.ToolRoundLimitExceeded {
                return error
            } catch {
                Issue.record("Expected the tool-round limit, got \(error)")
                return nil
            }
        }

        @Test(arguments: Provider.allCases)
        func streamedToolRoundsStopAtLimit(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            let limit = GenerationOptions().maximumToolRounds
            for round in 0 ... limit {
                UsageURLProtocol.enqueue(json: try provider.toolStream(city: "City \(round)", callID: "call_\(round)"))
            }
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            let exceeded = try #require(
                await Self.toolRoundLimitExceeded {
                    _ = try await session.streamResponse(to: "Weather?").collect()
                }
            )
            #expect(exceeded.limit == limit)
            // The rounds that ran are carried, each call followed by its output; the round past the limit isn't.
            #expect(Self.toolCallCount(in: exceeded.transcriptEntries) == limit)
            #expect(Self.toolOutputCount(in: exceeded.transcriptEntries) == limit)
            #expect(tool.cities.withLock { $0.count } == limit)
            #expect(UsageURLProtocol.recordedBodies.count == limit + 1)
            #expect(!session.isResponding)
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func toolRoundsStopAtLimit(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            var options = GenerationOptions()
            options.maximumToolRounds = 2
            for _ in 0 ... options.maximumToolRounds {
                UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "", tool: true)))
            }
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            let exceeded = try #require(
                await Self.toolRoundLimitExceeded {
                    _ = try await session.respond(to: "Weather?", options: options)
                }
            )
            #expect(exceeded.limit == 2)
            #expect(Self.toolCallCount(in: exceeded.transcriptEntries) == 2)
            #expect(Self.toolOutputCount(in: exceeded.transcriptEntries) == 2)
            #expect(tool.cities.withLock { $0.count } == 2)
        }

        private static func toolCallCount(in entries: ArraySlice<Transcript.Entry>) -> Int {
            entries.reduce(0) { count, entry in
                if case .toolCalls(let calls) = entry { count + calls.count } else { count }
            }
        }

        private static func toolOutputCount(in entries: ArraySlice<Transcript.Entry>) -> Int {
            entries.filter { if case .toolOutput = $0 { true } else { false } }.count
        }

        @Test(arguments: Provider.allCases)
        func streamedToolRoundsGoOnToTheirMaximum(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            let rounds = 10
            for round in 0 ..< rounds {
                UsageURLProtocol.enqueue(json: try provider.toolStream(city: "City \(round)", callID: "call_\(round)"))
            }
            UsageURLProtocol.enqueue(json: try provider.stream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            var options = GenerationOptions()
            options.maximumToolRounds = rounds
            _ = try await session.streamResponse(to: "Weather?", options: options).collect()
            #expect(tool.cities.withLock { $0.count } == rounds)
            #expect(UsageURLProtocol.recordedBodies.count == rounds + 1)
        }

        /// Calling the same tool with the same arguments again can be what's asked,
        /// so a round that repeats the previous one runs like any other.
        @Test(arguments: Provider.allCases)
        func streamedRepeatedToolRoundRuns(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            UsageURLProtocol.enqueue(json: try provider.toolStream(callID: "call_2"))
            UsageURLProtocol.enqueue(json: try provider.stream())
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            let response = try await session.streamResponse(to: "Weather?").collect()
            #expect(response.content == "Hello")
            #expect(tool.cities.withLock { $0 } == ["Paris", "Paris"])
            #expect(UsageURLProtocol.recordedBodies.count == 3)
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func repeatedToolRoundRuns(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "", tool: true)))
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "", tool: true)))
            UsageURLProtocol.enqueue(json: try Self.json(provider.response()))
            let tool = RecordingWeatherTool()
            let session = provider.makeSession(tools: [tool])
            let response = try await session.respond(to: "Weather?")
            #expect(response.content == "Hello")
            #expect(tool.cities.withLock { $0 } == ["Paris", "Paris"])
            #expect(UsageURLProtocol.recordedBodies.count == 3)
        }

        @Test(arguments: [Provider.responses, .openResponses])
        func responsesDecodeErrorObjects(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            var body = provider.response(counts: provider.counts)
            body["error"] = ["code": "server_error", "message": "The model failed."]
            UsageURLProtocol.enqueue(json: try Self.json(body))
            let response = try await provider.makeSession().respond(to: "Hi")
            #expect(response.content == "Hello")
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func toolRoundsAccumulateUsage(_ provider: Provider) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try Self.json(provider.response(text: "", counts: provider.counts, tool: true))
            )
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
    final class UsageURLProtocol: URLProtocol {
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
