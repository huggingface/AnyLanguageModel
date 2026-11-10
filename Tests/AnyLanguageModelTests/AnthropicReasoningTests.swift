import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Anthropic reasoning replay", .serialized)
    struct AnthropicReasoningTests {
        private func model() -> AnthropicLanguageModel {
            .init(apiKey: "fixture", model: "fixture", session: ReasoningURLProtocol.makeSession())
        }

        @Test(arguments: [false, true])
        func thinkingSurvivesPersistenceAndReplaysWithAnswer(streaming: Bool) async throws {
            ReasoningURLProtocol.reset()
            let session = LanguageModelSession(model: model())
            let result: LanguageModelSession.Response<String>
            if streaming {
                ReasoningURLProtocol.enqueue(eventStream: events())
                var ids: Set<String> = []
                var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
                for try await snapshot in session.streamResponse(to: "Question") {
                    snapshots.append(snapshot)
                    if let entry = snapshot.transcriptEntries.first { ids.insert(entry.id) }
                    let visible: String = snapshot.content
                    #expect(!visible.contains("Consider"))
                }
                #expect(ids.count == 1)
                #expect(snapshots.contains { $0.content.isEmpty && !$0.transcriptEntries.isEmpty })
                result = try await LanguageModelSession.ResponseStream<String>(
                    stream: AsyncThrowingStream {
                        $0.yield(snapshots.last!); $0.finish()
                    }
                ).collect()
            } else {
                ReasoningURLProtocol.enqueue(json: response())
                result = try await session.respond(to: "Question")
            }
            #expect(result.content == "Answer")
            let entry = try #require(result.transcriptEntries.first)
            guard case .reasoning(let reasoning) = entry else { Issue.record("Missing reasoning"); return }
            #expect(reasoning.segments.first?.description == "Consider")
            #expect(reasoning.signature == Data("opaque-signature".utf8))
            let restored = try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(session.transcript))
            ReasoningURLProtocol.enqueue(json: response())
            _ = try await LanguageModelSession(model: model(), transcript: restored).respond(to: "Again")
            let body = try #require(ReasoningURLProtocol.recordedBodies.last)
            let json = try JSONSerialization.jsonObject(with: body) as! [String: Any]
            let messages = json["messages"] as! [[String: Any]]
            let assistant = try #require(messages.first { $0["role"] as? String == "assistant" })
            let content = assistant["content"] as! [[String: Any]]
            #expect(content.map { $0["type"] as! String } == ["thinking", "text"])
            #expect(content[0]["signature"] as? String == "opaque-signature")
            #expect(content[1]["text"] as? String == "Answer")
        }

        @Test(arguments: [false, true])
        func thinkingAndToolRoundTextReplayInOrder(streaming: Bool) async throws {
            ReasoningURLProtocol.reset()
            let session = LanguageModelSession(model: model(), tools: [WeatherTool()])
            let result: LanguageModelSession.Response<String>
            if streaming {
                ReasoningURLProtocol.enqueue(eventStream: events(tool: true, withToolText: true))
                ReasoningURLProtocol.enqueue(eventStream: events())
                result = try await session.streamResponse(to: "Weather").collect()
            } else {
                ReasoningURLProtocol.enqueue(json: response(tool: true, withToolText: true))
                ReasoningURLProtocol.enqueue(json: response())
                result = try await session.respond(to: "Weather")
            }
            #expect(result.content == "Checking.Answer")
            #expect(result.transcriptEntries.count == 4)
            #expect(Set(result.transcriptEntries.map(\.id)).count == result.transcriptEntries.count)
            try #require(ReasoningURLProtocol.recordedBodies.count == 2)
            let body = String(decoding: ReasoningURLProtocol.recordedBodies[1], as: UTF8.self)
            #expect(body.contains("opaque-signature"))
            #expect(body.contains("tool_result"))

            let restored = try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(session.transcript))
            ReasoningURLProtocol.enqueue(json: response())
            _ = try await LanguageModelSession(model: model(), transcript: restored).respond(to: "Again")
            let request = try #require(ReasoningURLProtocol.recordedBodies.last)
            let json = try #require(try JSONSerialization.jsonObject(with: request) as? [String: Any])
            let messages = try #require(json["messages"] as? [[String: Any]])
            try #require(messages.count == 5)
            #expect(
                messages.compactMap { $0["role"] as? String } == ["user", "assistant", "user", "assistant", "user"]
            )
            let beforeTool = try #require(messages[1]["content"] as? [[String: Any]])
            #expect(beforeTool.compactMap { $0["type"] as? String } == ["thinking", "text", "tool_use"])
            #expect(beforeTool.first?["signature"] as? String == "opaque-signature")
            #expect(beforeTool.compactMap { $0["text"] as? String } == ["Checking."])
            let outputs = try #require(messages[2]["content"] as? [[String: Any]])
            #expect(outputs.first?["type"] as? String == "tool_result")
            #expect(outputs.first?["tool_use_id"] as? String == "call")
            let afterTool = try #require(messages[3]["content"] as? [[String: Any]])
            #expect(afterTool.compactMap { $0["text"] as? String } == ["Answer"])
        }

        enum LimitedToolFlow: CaseIterable {
            case streaming, profileStreaming, profileResponse
        }

        /// Hands over one message, once.
        private final class SteeringDelegate: ToolExecutionDelegate, @unchecked Sendable {
            private let lock = NSLock()
            private var pending = [AnyLanguageModel.Prompt("Skip the umbrella advice.")]

            func pendingMessages(in session: LanguageModelSession) async -> [AnyLanguageModel.Prompt] {
                lock.withLock {
                    defer { pending = [] }
                    return pending
                }
            }
        }

        @Test(arguments: [false, true])
        func pendingMessagesJoinTheResponseAfterTheToolRound(streaming: Bool) async throws {
            ReasoningURLProtocol.reset()
            if streaming {
                ReasoningURLProtocol.enqueue(eventStream: events(tool: true))
                ReasoningURLProtocol.enqueue(eventStream: events())
                ReasoningURLProtocol.enqueue(eventStream: events())
            } else {
                ReasoningURLProtocol.enqueue(json: response(tool: true))
                ReasoningURLProtocol.enqueue(json: response())
                ReasoningURLProtocol.enqueue(json: response())
            }
            let session = LanguageModelSession(model: model(), tools: [WeatherTool()])
            session.toolExecutionDelegate = SteeringDelegate()
            if streaming {
                _ = try await session.streamResponse(to: "Weather").collect()
                _ = try await session.streamResponse(to: "And tomorrow?").collect()
            } else {
                _ = try await session.respond(to: "Weather")
                _ = try await session.respond(to: "And tomorrow?")
            }

            let bodies = ReasoningURLProtocol.recordedBodies
            try #require(bodies.count == 3)
            func messages(_ body: Data) throws -> [[String: Any]] {
                let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
                return try #require(json["messages"] as? [[String: Any]])
            }
            // After the round's tool results, the message, as its own user message.
            let inFlight = try messages(bodies[1])
            let note = try #require(inFlight.last)
            #expect(note["role"] as? String == "user")
            #expect(String(describing: note["content"] ?? "").contains("Skip the umbrella advice."))
            #expect(String(describing: inFlight[inFlight.count - 2]["content"] ?? "").contains("tool_result"))
            // The next turn replays it from the transcript as it was sent.
            let replayed = try messages(bodies[2])
            let replayedNote = try #require(
                replayed.first { String(describing: $0["content"] ?? "").contains("Skip the umbrella advice.") }
            )
            #expect(NSDictionary(dictionary: note).isEqual(to: replayedNote))
        }

        /// The round past the limit didn't run, so its reasoning isn't carried with the rounds that did.
        @Test(arguments: LimitedToolFlow.allCases)
        func toolRoundLimitOmitsTheReasoningOfTheRoundPastIt(flow: LimitedToolFlow) async throws {
            ReasoningURLProtocol.reset()
            let streaming = flow != .profileResponse
            if streaming {
                ReasoningURLProtocol.enqueue(eventStream: events(tool: true))
                ReasoningURLProtocol.enqueue(eventStream: events(tool: true))
            } else {
                ReasoningURLProtocol.enqueue(json: response(tool: true))
                ReasoningURLProtocol.enqueue(json: response(tool: true))
            }
            var options = GenerationOptions()
            options.maximumToolRounds = 1
            let session =
                flow == .streaming
                ? LanguageModelSession(model: model(), tools: [WeatherTool()])
                : LanguageModelSession(profile: LanguageModelSession.Profile { WeatherTool() }.model(model()))
            do {
                if streaming {
                    _ = try await session.streamResponse(to: "Weather", options: options).collect()
                } else {
                    _ = try await session.respond(to: "Weather", options: options)
                }
                Issue.record("Expected the tool-round limit")
            } catch let error as LanguageModelSession.ToolRoundLimitExceeded {
                let kinds = error.transcriptEntries.map { entry in
                    switch entry {
                    case .reasoning: "reasoning"
                    case .toolCalls: "toolCalls"
                    case .toolOutput: "toolOutput"
                    default: "other"
                    }
                }
                #expect(kinds == ["reasoning", "toolCalls", "toolOutput"])
            }
        }

        @Test(arguments: [false, true])
        func redactedThinkingNeverBecomesDisplayText(streaming: Bool) async throws {
            ReasoningURLProtocol.reset()
            let session = LanguageModelSession(model: model())
            if streaming {
                ReasoningURLProtocol.enqueue(eventStream: [
                    #"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque-redacted"}}"#,
                    #"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
                    #"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}"#,
                    #"{"type":"message_stop"}"#,
                ])
                _ = try await session.streamResponse(to: "Question").collect()
            } else {
                ReasoningURLProtocol.enqueue(
                    json:
                        #"{"id":"msg","type":"message","role":"assistant","model":"fixture","content":[{"type":"redacted_thinking","data":"opaque-redacted"},{"type":"text","text":"Answer"}],"stop_reason":"end_turn"}"#
                )
                _ = try await session.respond(to: "Question")
            }
            guard case .reasoning(let reasoning) = session.transcript[1] else {
                Issue.record("Missing redacted state"); return
            }
            #expect(reasoning.segments.isEmpty)
            #expect(reasoning.signature == Data("opaque-redacted".utf8))
            ReasoningURLProtocol.enqueue(json: response())
            _ = try await session.respond(to: "Next")
            let json =
                try JSONSerialization.jsonObject(with: ReasoningURLProtocol.recordedBodies.last!) as! [String: Any]
            let messages = json["messages"] as! [[String: Any]]
            let blocks = messages.first { $0["role"] as? String == "assistant" }!["content"] as! [[String: Any]]
            #expect(blocks.first?["type"] as? String == "redacted_thinking")
            #expect(blocks.first?["data"] as? String == "opaque-redacted")
        }

        @Test(arguments: [false, true], [false, true])
        func foreignReasoningIsSkippedWithoutChangingHistory(streaming: Bool, missingProvider: Bool) async throws {
            ReasoningURLProtocol.reset()
            let original = Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "Earlier question"))])),
                .reasoning(
                    .init(
                        metadata: missingProvider ? [:] : ["provider": GeneratedContent("other")],
                        segments: [.text(.init(content: "Foreign reasoning"))],
                        signature: Data("foreign-signature".utf8)
                    )
                ),
                .response(.init(assetIDs: [], segments: [.text(.init(content: "Earlier answer"))])),
            ])
            let restored = try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(original))
            let session = LanguageModelSession(model: model(), transcript: restored)
            if streaming {
                ReasoningURLProtocol.enqueue(eventStream: events())
            } else {
                ReasoningURLProtocol.enqueue(json: response())
            }
            let result =
                try await streaming ? session.streamResponse(to: "Next").collect() : session.respond(to: "Next")
            #expect(result.content == "Answer")
            #expect(ReasoningURLProtocol.recordedBodies.count == 1)
            let body = String(decoding: try #require(ReasoningURLProtocol.recordedBodies.first), as: UTF8.self)
            #expect(body.contains("Earlier question"))
            #expect(body.contains("Earlier answer"))
            #expect(!body.contains("Foreign reasoning"))
            #expect(!body.contains("foreign-signature"))
            #expect(Array(session.transcript.prefix(original.count)) == Array(original))
        }

        @Test(arguments: ["chat", "responses", "open-responses", "gemini", "ollama"], [false, true])
        func otherProviderRequestOmitsReasoningAndPreservesCodableHistory(provider: String, streaming: Bool)
            async throws
        {
            ReasoningURLProtocol.reset()
            let original = Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "Earlier question"))])),
                .reasoning(
                    .init(
                        metadata: ["provider": GeneratedContent("anthropic")],
                        segments: [.text(.init(content: "Private reasoning"))],
                        signature: Data("opaque-secret".utf8)
                    )
                ),
                .response(.init(assetIDs: [], segments: [.text(.init(content: "Earlier answer"))])),
            ])
            let restored = try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(original))
            let transport = ReasoningURLProtocol.makeSession()
            let endpoint = URL(string: "https://fixture.invalid/v1/")!
            let providerModel: any LanguageModel
            switch provider {
            case "chat", "responses":
                providerModel = OpenAILanguageModel(
                    baseURL: endpoint,
                    apiKey: "fixture",
                    model: "fixture",
                    apiVariant: provider == "chat" ? .chatCompletions : .responses,
                    session: transport
                )
            case "open-responses":
                providerModel = OpenResponsesLanguageModel(
                    baseURL: endpoint,
                    apiKey: "fixture",
                    model: "fixture",
                    session: transport
                )
            case "gemini":
                providerModel = GeminiLanguageModel(
                    baseURL: endpoint,
                    apiKey: "fixture",
                    model: "fixture",
                    session: transport
                )
            default: providerModel = OllamaLanguageModel(baseURL: endpoint, model: "fixture", session: transport)
            }
            let session = LanguageModelSession(model: providerModel, transcript: restored)
            // A fixed HTTP failure isolates request projection from each provider's response decoder.
            ReasoningURLProtocol.enqueue(json: #"{"error":{"message":"fixture rejection"}}"#, statusCode: 418)
            do {
                if streaming {
                    _ = try await session.streamResponse(to: "Next").collect()
                } else {
                    _ = try await session.respond(to: "Next")
                }
                Issue.record("Expected fixture HTTP failure")
            } catch {
                #expect(!(error is Transcript.ReasoningReplayError))
            }
            #expect(ReasoningURLProtocol.recordedBodies.count == 1)
            let body = String(decoding: try #require(ReasoningURLProtocol.recordedBodies.first), as: UTF8.self)
            #expect(body.contains("Next"))
            // Compare against this adapter's existing history projection. Some adapters
            // intentionally project only the prompt or structured response metadata.
            let withoutReasoning = Transcript(
                entries: original.filter {
                    if case .reasoning = $0 { return false }
                    return true
                }
            )
            let baseline = LanguageModelSession(model: providerModel, transcript: withoutReasoning)
            ReasoningURLProtocol.enqueue(json: #"{"error":{"message":"fixture rejection"}}"#, statusCode: 418)
            do {
                if streaming {
                    _ = try await baseline.streamResponse(to: "Next").collect()
                } else {
                    _ = try await baseline.respond(to: "Next")
                }
                Issue.record("Expected fixture HTTP failure")
            } catch {}
            #expect(ReasoningURLProtocol.recordedBodies.count == 2)
            let baselineBody = try #require(ReasoningURLProtocol.recordedBodies.last)
            let actualJSON = try JSONSerialization.jsonObject(with: Data(body.utf8)) as! NSDictionary
            let baselineJSON = try JSONSerialization.jsonObject(with: baselineBody) as! NSDictionary
            #expect(actualJSON == baselineJSON)
            #expect(!body.contains("Private reasoning"))
            #expect(!body.contains("opaque-secret"))
            #expect(Array(session.transcript.prefix(original.count)) == Array(original))
            let saved = try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(session.transcript))
            #expect(Array(saved.prefix(original.count)) == Array(original))
        }

        @Test(arguments: [false, true])
        func nativeReasoningStillRejectsMissingSignature(streaming: Bool) async throws {
            ReasoningURLProtocol.reset()
            let transcript = Transcript(entries: [
                .reasoning(
                    .init(
                        metadata: ["provider": GeneratedContent("anthropic")],
                        segments: [.text(.init(content: "Native reasoning"))]
                    )
                )
            ])
            let session = LanguageModelSession(model: model(), transcript: transcript)
            await #expect(throws: Transcript.ReasoningReplayError.invalidSignature) {
                if streaming {
                    _ = try await session.streamResponse(to: "Next").collect()
                } else {
                    _ = try await session.respond(to: "Next")
                }
            }
            #expect(ReasoningURLProtocol.recordedBodies.isEmpty)
            #expect(Array(session.transcript.prefix(transcript.count)) == Array(transcript))
        }

        private func response(tool: Bool = false, withToolText: Bool = false) -> String {
            let last =
                tool
                ? #"{"type":"tool_use","id":"call","name":"getWeather","input":{"city":"Paris"}}"#
                : #"{"type":"text","text":"Answer"}"#
            return
                #"{"id":"msg","type":"message","role":"assistant","model":"fixture","content":[{"type":"thinking","thinking":"Consider","signature":"opaque-signature"},"#
                + (withToolText ? #"{"type":"text","text":"Checking."},"# : "")
                + last + #"],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}"#
        }

        private func events(tool: Bool = false, withToolText: Bool = false) -> [String] {
            var result = [
                #"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
                #"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Consider"}}"#,
                #"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"opaque-signature"}}"#,
                #"{"type":"content_block_stop","index":0}"#,
            ]
            if tool {
                if withToolText {
                    result.append(
                        #"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#
                    )
                    result.append(
                        #"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking."}}"#
                    )
                    result.append(#"{"type":"content_block_stop","index":1}"#)
                }
                result.append(
                    #"{"type":"content_block_start","index":\#(withToolText ? 2 : 1),"content_block":{"type":"tool_use","id":"call","name":"getWeather","input":{"city":"Paris"}}}"#
                )
            } else {
                result.append(#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#)
                result.append(
                    #"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}"#
                )
            }
            result.append(#"{"type":"message_stop"}"#)
            return result
        }
    }
#endif

#if canImport(Darwin) && !canImport(AsyncHTTPClient)

    /// A `URLProtocol` that answers requests from a queue of canned responses and records
    /// every request body it sees, so request/response round trips can be asserted offline.
    final class ReasoningURLProtocol: URLProtocol {
        struct Exchange: Sendable {
            var statusCode: Int = 200
            var body: Data
            var contentType: String = "application/json"
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

        /// Queues one JSON response, returned to the next request that arrives.
        static func enqueue(json: String, statusCode: Int = 200) {
            state.withLock { $0.pending.append(Exchange(statusCode: statusCode, body: Data(json.utf8))) }
        }

        static func enqueue(eventStream: [String]) {
            let body = eventStream.map { "data: \($0)\n\n" }.joined()
            state.withLock {
                $0.pending.append(Exchange(body: Data(body.utf8), contentType: "text/event-stream"))
            }
        }

        /// The bodies of the requests seen so far, in order.
        static var recordedBodies: [Data] {
            state.withLock { $0.recordedBodies }
        }

        /// A session that routes every request to this protocol.
        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [ReasoningURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            // URLSession moves `httpBody` to `httpBodyStream` before the protocol sees the request.
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
                headerFields: ["Content-Type": exchange.contentType]
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
