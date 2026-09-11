import Foundation
import Testing

@testable import AnyLanguageModel

#if LiteRT && (os(iOS) || os(macOS)) && !targetEnvironment(macCatalyst)
    @preconcurrency import LiteRTLM

    @Suite("LiteRTLanguageModel behavior", .timeLimit(.minutes(1)))
    struct LiteRTLanguageModelBehaviorTests {
        @Test(arguments: ["plain", "fenced", "prose"], [false, true])
        func generatesArraysAndScalars(wrapper: String, streaming: Bool) async throws {
            func check<Value: Generable & Equatable>(_ json: String, equals expected: Value) async throws {
                let reply: String
                switch wrapper {
                case "fenced": reply = "```json\n\(json)\n```"
                case "prose": reply = "The result is:\n\(json)\nThat is the answer."
                default: reply = json
                }
                let conversation = StubLiteRTConversation(chunks: reply.map(String.init))
                let runtime = StubLiteRTRuntime([conversation])
                let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }))
                if streaming {
                    var values: [Value] = []
                    for try await snapshot in session.streamResponse(to: "Generate", generating: Value.self) {
                        values.append(try Value(snapshot.rawContent))
                    }
                    #expect(values == [expected])
                } else {
                    #expect(try await session.respond(to: "Generate", generating: Value.self).content == expected)
                }
                #expect(conversation.requests.withLock { $0.first?.text.contains("JSON value") } == true)
            }
            try await check(#"{"answer":"hello"}"#, equals: LiteRTTestAnswer(answer: "hello"))
            try await check("[]", equals: [Int]())
            try await check("[1,2,3]", equals: [1, 2, 3])
            try await check("[[1,2],[],[3]]", equals: [[1, 2], [], [3]])
            try await check("42", equals: 42)
            try await check("-1.25e+2", equals: -125.0)
            try await check("true", equals: true)
            try await check("false", equals: false)
            try await check(#"["a]b", "a\"b", "a\\b"]"#, equals: ["a]b", "a\"b", "a\\b"])
        }

        @Test(arguments: ["[1,", "[1:2]", "1e", "tru", "falsehood"], [false, true])
        func rejectsIncompleteOrInvalidStructuredResponses(reply: String, streaming: Bool) async throws {
            let runtime = StubLiteRTRuntime([StubLiteRTConversation(reply: reply)])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }))
            await #expect {
                if streaming {
                    for try await _ in session.streamResponse(to: "Generate", generating: [Int].self) {
                        Issue.record("Incomplete JSON must not produce a snapshot")
                    }
                } else {
                    _ = try await session.respond(to: "Generate", generating: [Int].self)
                }
            } throws: { error in
                guard case LanguageModelSession.GenerationError.decodingFailure = error else { return false }
                return true
            }
        }

        @Test(arguments: [false, true], [UInt64(123), UInt64(Int32.max) + 1, UInt64.max])
        func preservesExplicitSamplingModesAndSeeds(streaming: Bool, seed: UInt64) async throws {
            for sampling in [
                GenerationOptions.SamplingMode.random(top: 10, seed: seed),
                .random(probabilityThreshold: 0.8, seed: seed),
            ] {
                let runtime = StubLiteRTRuntime([StubLiteRTConversation(reply: "Hello")])
                let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }))
                let options = GenerationOptions(sampling: sampling, temperature: 0.7)
                if streaming {
                    for try await _ in session.streamResponse(to: "Hello", options: options) {}
                } else {
                    _ = try await session.respond(to: "Hello", options: options)
                }
                let config = try #require(runtime.configurations.withLock { $0.first?.sampler })
                switch sampling.mode {
                case .topK:
                    #expect(config.topK == 10)
                    #expect(config.topP == 1.0)
                case .nucleus:
                    #expect(config.topK == Int(Int32.max))
                    #expect(config.topP == 0.8)
                case .greedy: Issue.record("Unexpected sampling mode")
                }
                #expect(config.temperature == 0.7)
                let expectedSeed: Int = seed == 123 ? 123 : (seed == UInt64.max ? -1 : Int(Int32.min))
                #expect(config.seed == expectedSeed)
            }
        }

        @Test(arguments: [false, true])
        func forwardsResponseTokenLimit(streaming: Bool) async throws {
            let conversation = StubLiteRTConversation(reply: "Hello")
            let runtime = StubLiteRTRuntime([conversation])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }))
            let options = GenerationOptions(maximumResponseTokens: 7)
            if streaming {
                for try await _ in session.streamResponse(to: "Hello", options: options) {}
            } else {
                _ = try await session.respond(to: "Hello", options: options)
            }
            #expect(conversation.requests.withLock { $0.map(\.maxOutputTokens) } == [7])
            #expect(conversation.cancelCount.withLock { $0 } == 0)
        }

        @Test(arguments: [false, true])
        func cancelsNativeInference(streaming: Bool) async throws {
            let conversation = StubLiteRTConversation(reply: "Hello", keepRunning: true)
            let runtime = StubLiteRTRuntime([conversation])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }))
            let task = Task {
                if streaming {
                    for try await _ in session.streamResponse(to: "Hello") {}
                } else {
                    _ = try await session.respond(to: "Hello")
                }
            }
            await conversation.started.wait()
            task.cancel()
            if streaming {
                do { try await task.value } catch { #expect(error is CancellationError) }
            } else {
                await #expect(throws: CancellationError.self) { try await task.value }
            }
            await conversation.cancelled.wait()
            #expect(conversation.cancelCount.withLock { $0 } == 1)
        }

        @Test func abandoningStreamCancelsNativeInference() async throws {
            let conversation = StubLiteRTConversation(reply: "Hello", keepRunning: true)
            let runtime = StubLiteRTRuntime([conversation])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }))
            for try await _ in session.streamResponse(to: "Hello") { break }
            await conversation.cancelled.wait()
            #expect(conversation.cancelCount.withLock { $0 } == 1)
        }

        @Test func cancellationDuringLoadingDoesNotStartInference() async throws {
            let started = LiteRTTestSignal()
            let release = LiteRTTestSignal()
            let runtime = StubLiteRTRuntime([StubLiteRTConversation(reply: "Hello")])
            let model = LiteRTLanguageModel(load: {
                await started.signal()
                await release.wait()
                return runtime
            })
            let session = LanguageModelSession(model: model)
            let task = Task { try await session.respond(to: "Hello") }
            await started.wait()
            task.cancel()
            await release.signal()
            await #expect(throws: CancellationError.self) { try await task.value }
            #expect(runtime.configurations.withLock { $0.isEmpty })
            _ = try await session.respond(to: "Try again")
            #expect(runtime.configurations.withLock { $0.count } == 1)
        }

        @Test func retriesFailedLoadingAndSharesSuccessfulRuntime() async throws {
            let loads = Locked(0)
            let runtime = StubLiteRTRuntime([
                StubLiteRTConversation(reply: "Hello"),
                StubLiteRTConversation(reply: "Again"),
            ])
            let model = LiteRTLanguageModel(load: {
                let count = loads.withLock {
                    $0 += 1; return $0
                }
                if count == 1 { throw URLError(.timedOut) }
                return runtime
            })
            let session = LanguageModelSession(model: model)
            await #expect(throws: URLError.self) { try await session.respond(to: "Hello") }
            #expect(try await session.respond(to: "Retry").content == "Hello")
            #expect(try await session.respond(to: "Again").content == "Again")
            #expect(loads.withLock { $0 } == 2)
        }

        @Test func concurrentWaitersCannotDiscardANewerRetry() async throws {
            let loads = Locked(0)
            let started = LiteRTTestSignal()
            let release = LiteRTTestSignal()
            let runtime = StubLiteRTRuntime([])
            let loader = LiteRTModelLoader {
                let count = loads.withLock {
                    $0 += 1; return $0
                }
                if count == 1 {
                    await started.signal()
                    await release.wait()
                    throw URLError(.timedOut)
                }
                return runtime
            }
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0 ..< 32 {
                    group.addTask {
                        _ = try? await loader.ready()
                        _ = try await loader.ready()
                    }
                }
                await started.wait()
                await release.signal()
                try await group.waitForAll()
            }
            #expect(loads.withLock { $0 } == 2)
        }

        @Test(arguments: [#"{"answer":42}"#, "[1,2]", "42", "true", #""hello""#])
        func preservesStructuredToolOutputAndTranscript(json: String) async throws {
            let output = try GeneratedContent(json: json)
            let tool = LiteRTTestTool(output: output)
            let first = StubLiteRTConversation(reply: toolCall)
            let second = StubLiteRTConversation(reply: "Done")
            let third = StubLiteRTConversation(reply: "Remembered")
            let runtime = StubLiteRTRuntime([first, second, third])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }), tools: [tool])
            _ = try await session.respond(to: "Use the tool", options: .init(maximumResponseTokens: 9))
            let nextPrompt = try #require(second.requests.withLock { $0.first?.text })
            #expect(nextPrompt.contains(output.jsonString))
            #expect(second.requests.withLock { $0.first?.maxOutputTokens } == 9)
            _ = try await session.respond(to: "What did the tool return?")
            let history = try #require(runtime.configurations.withLock { $0.last?.history })
            #expect(history.contains { $0.contains(output.jsonString) })
        }

        @Test func stoppedStructuredResponseThrowsInsteadOfTrapping() async throws {
            let tool = LiteRTTestTool(output: GeneratedContent("unused"))
            let runtime = StubLiteRTRuntime([StubLiteRTConversation(reply: toolCall)])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }), tools: [tool])
            session.toolExecutionDelegate = LiteRTStopDelegate()
            await #expect {
                try await session.respond(to: "Use the tool", generating: LiteRTTestAnswer.self)
            } throws: { error in
                guard case LanguageModelSession.GenerationError.decodingFailure = error else { return false }
                return true
            }
            #expect(tool.calls.withLock { $0 } == 0)
        }

        @Test func stoppedTextResponsePreservesPendingToolCall() async throws {
            let tool = LiteRTTestTool(output: GeneratedContent("unused"))
            let runtime = StubLiteRTRuntime([StubLiteRTConversation(reply: toolCall)])
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }), tools: [tool])
            session.toolExecutionDelegate = LiteRTStopDelegate()
            #expect(try await session.respond(to: "Use the tool").content.isEmpty)
            #expect(
                session.transcript.contains {
                    if case .toolCalls = $0 { return true }; return false
                }
            )
            #expect(tool.calls.withLock { $0 } == 0)
        }

        @Test func rejectsUnresolvedToolCallAtIterationLimit() async throws {
            let tool = LiteRTTestTool(output: GeneratedContent("value"))
            let runtime = StubLiteRTRuntime((0 ..< 5).map { _ in StubLiteRTConversation(reply: toolCall) })
            let session = LanguageModelSession(model: LiteRTLanguageModel(load: { runtime }), tools: [tool])
            await #expect {
                try await session.respond(to: "Keep calling the tool")
            } throws: { error in
                guard case LanguageModelSession.GenerationError.decodingFailure = error else { return false }
                return true
            }
            #expect(tool.calls.withLock { $0 } == 4)
        }

        @Test func omitsHiddenToolSchemaButStillExecutesItsCalls() async throws {
            let tool = LiteRTTestTool(output: GeneratedContent("value"), includesSchemaInInstructions: false)
            let runtime = StubLiteRTRuntime([
                StubLiteRTConversation(reply: toolCall), StubLiteRTConversation(reply: "Done"),
            ])
            let session = LanguageModelSession(
                model: LiteRTLanguageModel(load: { runtime }),
                tools: [tool],
                instructions: "Help the user"
            )
            _ = try await session.respond(to: "Use your built-in tool")
            #expect(runtime.configurations.withLock { $0.first?.systemMessage } == "Help the user")
            #expect(tool.calls.withLock { $0 } == 1)
        }

        @Test(arguments: ["success", "missing", "network"], [false, true])
        func remoteImagesUseAsyncDownloadsAndPropagateErrors(path: String, streaming: Bool) async throws {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [LiteRTImageURLProtocol.self]
            let imageSession = URLSession(configuration: configuration)
            defer { imageSession.invalidateAndCancel() }
            let conversation = StubLiteRTConversation(reply: "An image")
            let runtime = StubLiteRTRuntime([conversation])
            let model = LiteRTLanguageModel(load: { runtime }, imageSession: imageSession)
            let session = LanguageModelSession(model: model)
            let image = Transcript.ImageSegment(url: URL(string: "https://images.invalid/\(path)")!)
            do {
                if streaming {
                    for try await _ in session.streamResponse(to: "Describe", images: [image]) {}
                } else {
                    _ = try await session.respond(to: "Describe", images: [image])
                }
                #expect(path == "success")
                #expect(conversation.requests.withLock { $0.first?.images } == [Data("image bytes".utf8)])
            } catch let error as URLError {
                #expect(error.code == (path == "missing" ? .badServerResponse : .notConnectedToInternet))
                #expect(path != "success")
                #expect(conversation.requests.withLock { $0.isEmpty })
            }
        }
    }

    private let toolCall = #"{"tool_call":{"name":"lookup","arguments":{"query":"answer"}}}"#

    @Generable
    private struct LiteRTTestAnswer: Equatable {
        var answer: String
    }

    private struct LiteRTTestTool: AnyLanguageModel.Tool {
        let name = "lookup"
        let description = "Looks up an answer"
        let output: GeneratedContent
        var includesSchemaInInstructions = true
        let calls = Locked(0)

        @Generable
        struct Arguments {
            var query: String
        }

        func call(arguments: Arguments) async throws -> GeneratedContent {
            calls.withLock { $0 += 1 }
            return output
        }
    }

    private struct LiteRTStopDelegate: ToolExecutionDelegate {
        func toolCallDecision(
            for toolCall: Transcript.ToolCall,
            in session: LanguageModelSession
        ) async -> ToolExecutionDecision { .stop }
    }

    private actor LiteRTTestSignal {
        private var signalled = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func wait() async {
            if signalled { return }
            await withCheckedContinuation { waiters.append($0) }
        }

        func signal() {
            signalled = true
            for waiter in waiters { waiter.resume() }
            waiters = []
        }
    }

    private final class StubLiteRTRuntime: LiteRTRuntime {
        struct Sampler: Sendable {
            var topK: Int
            var topP: Float
            var temperature: Float
            var seed: Int
        }
        struct Configuration: Sendable {
            var sampler: Sampler?
            var systemMessage: String?
            var history: [String]
        }
        let configurations = Locked<[Configuration]>([])
        let conversations: [StubLiteRTConversation]

        init(_ conversations: [StubLiteRTConversation]) {
            self.conversations = conversations
        }

        func makeConversation(config: ConversationConfig) async throws -> any LiteRTConversation {
            let index = configurations.withLock { records in
                let index = records.count
                records.append(
                    Configuration(
                        sampler: config.samplerConfig.map {
                            Sampler(topK: $0.topK, topP: $0.topP, temperature: $0.temperature, seed: $0.seed)
                        },
                        systemMessage: config.systemMessage?.toString,
                        history: config.initialMessages.map(\.toString)
                    )
                )
                return index
            }
            guard conversations.indices.contains(index) else { throw URLError(.resourceUnavailable) }
            return conversations[index]
        }
    }

    private final class StubLiteRTConversation: LiteRTConversation {
        struct Request: Sendable {
            var text: String
            var images: [Data]
            var maxOutputTokens: Int?
        }
        let chunks: [String]
        let keepRunning: Bool
        let requests = Locked<[Request]>([])
        let cancelCount = Locked(0)
        let continuation = Locked<AsyncThrowingStream<Message, Error>.Continuation?>(nil)
        let started = LiteRTTestSignal()
        let cancelled = LiteRTTestSignal()

        convenience init(reply: String, keepRunning: Bool = false) {
            self.init(chunks: [reply], keepRunning: keepRunning)
        }

        init(chunks: [String], keepRunning: Bool = false) {
            self.chunks = chunks
            self.keepRunning = keepRunning
        }

        func sendMessageStream(_ message: Message, maxOutputTokens: Int?) -> AsyncThrowingStream<Message, Error> {
            requests.withLock {
                $0.append(
                    Request(
                        text: message.toString,
                        images: message.contents.compactMap {
                            if case .imageData(let data) = $0 { return data }; return nil
                        },
                        maxOutputTokens: maxOutputTokens
                    )
                )
            }
            return AsyncThrowingStream { continuation in
                self.continuation.withLock { $0 = continuation }
                for chunk in chunks { continuation.yield(Message(chunk, role: .model)) }
                if !keepRunning { continuation.finish() }
                Task { await started.signal() }
            }
        }

        func cancel() {
            cancelCount.withLock { $0 += 1 }
            continuation.withLock { $0 }?.finish(throwing: CancellationError())
            Task { await cancelled.signal() }
        }
    }

    private final class LiteRTImageURLProtocol: URLProtocol {
        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            let url = request.url!
            if url.lastPathComponent == "network" {
                client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
                return
            }
            let response = HTTPURLResponse(
                url: url,
                statusCode: url.lastPathComponent == "missing" ? 404 : 200,
                httpVersion: "HTTP/1.1",
                headerFields: nil
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data("image bytes".utf8))
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }
#endif
