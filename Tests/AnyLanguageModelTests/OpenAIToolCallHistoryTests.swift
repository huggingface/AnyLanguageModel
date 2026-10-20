import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("OpenAI tool call history", .serialized)
    struct OpenAIToolCallHistoryTests {
        private let chatCompletion = """
            {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "test",
             "choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": "Done."}}]}
            """

        private let responsesResponse = """
            {"id": "resp-1", "object": "response", "created_at": 0, "model": "test", "status": "completed",
             "output": [{"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]}
            """

        private func transcript() throws -> Transcript {
            let call = Transcript.ToolCall(
                id: "call-1",
                toolName: "getWeather",
                arguments: try GeneratedContent(json: #"{"city":"Paris"}"#)
            )
            return Transcript(entries: [
                .prompt(Transcript.Prompt(segments: [.text(.init(content: "Weather in Paris?"))])),
                .toolCalls(Transcript.ToolCalls([call])),
                .toolOutput(
                    Transcript.ToolOutput(
                        id: "call-1",
                        toolName: "getWeather",
                        segments: [.text(.init(content: "Sunny"))]
                    )
                ),
                .response(Transcript.Response(assetIDs: [], segments: [.text(.init(content: "It's sunny."))])),
            ])
        }

        /// The arguments a replayed tool call carries, parsed back from the request body.
        private func replayedArguments(in body: Data) throws -> [String] {
            let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            if let messages = json["messages"] as? [[String: Any]] {
                return messages.flatMap { message in
                    (message["tool_calls"] as? [[String: Any]] ?? []).compactMap {
                        ($0["function"] as? [String: Any])?["arguments"] as? String
                    }
                }
            }
            let input = try #require(json["input"] as? [[String: Any]])
            return input.filter { $0["type"] as? String == "function_call" }.compactMap { $0["arguments"] as? String }
        }

        @Test(arguments: [OpenAILanguageModel.APIVariant.chatCompletions, .responses])
        func replayedToolCallArgumentsArePlainJSON(variant: OpenAILanguageModel.APIVariant) async throws {
            ToolCallHistoryURLProtocol.reset(
                responses: [variant == .chatCompletions ? chatCompletion : responsesResponse]
            )
            let model = OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                apiVariant: variant,
                session: ToolCallHistoryURLProtocol.makeSession()
            )
            let session = LanguageModelSession(model: model, transcript: try transcript())
            _ = try await session.respond(to: "And tomorrow?")

            let body = try #require(ToolCallHistoryURLProtocol.recordedBodies.last)
            let arguments = try replayedArguments(in: body)
            #expect(arguments.count == 1)
            let decoded = try #require(
                arguments.first.flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: String] }
            )
            #expect(decoded == ["city": "Paris"])
        }

        private let toolCallArguments = #"{\"unit\": \"celsius\", \"city\": \"Paris\"}"#

        private func completion(_ text: String) -> String {
            """
            {"id": "c", "object": "chat.completion", "created": 0, "model": "test",
             "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "\(text)"}}]}
            """
        }

        private func toolCallCompletion(content: String) -> String {
            let contentJSON = content.isEmpty ? "null" : "\"\(content)\""
            return """
                {"id": "c", "object": "chat.completion", "created": 0, "model": "test",
                 "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": \(contentJSON),
                   "tool_calls": [{"id": "call-1", "type": "function",
                                   "function": {"name": "getWeather", "arguments": "\(toolCallArguments)"}}]}}]}
                """
        }

        private func chunk(_ delta: String, finish: String = "null") -> String {
            #"data: {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "test", "choices": [{"index": 0, "delta": \#(delta), "finish_reason": \#(finish)}]}"#
                + "\n\n"
        }

        private func toolCallStream(content: String) -> String {
            let contentJSON = content.isEmpty ? "" : #", "content": "\#(content)""#
            return chunk(
                #"{"role": "assistant"\#(contentJSON), "tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {"name": "getWeather", "arguments": "\#(toolCallArguments)"}}]}"#
            ) + chunk("{}", finish: #""tool_calls""#) + "data: [DONE]\n\n"
        }

        private func textStream(_ text: String) -> String {
            chunk(#"{"role": "assistant", "content": "\#(text)"}"#) + chunk("{}", finish: #""stop""#)
                + "data: [DONE]\n\n"
        }

        /// The assistant message carrying the tool call, as sent in a request body.
        private func toolCallMessage(in body: Data) throws -> [String: Any] {
            let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            let messages = try #require(json["messages"] as? [[String: Any]])
            return try #require(messages.first { $0["tool_calls"] != nil })
        }

        @Test(arguments: [false, true], ["", "Let me check."])
        func toolCallMessageIsTheSameInFlightAndInHistory(streaming: Bool, content: String) async throws {
            ToolCallHistoryURLProtocol.reset(
                responses: streaming
                    ? [toolCallStream(content: content), textStream("Sunny."), textStream("Done.")]
                    : [toolCallCompletion(content: content), completion("Sunny."), completion("Done.")]
            )
            let model = OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                session: ToolCallHistoryURLProtocol.makeSession()
            )
            let session = LanguageModelSession(model: model, tools: [ReorderedWeatherTool()])
            if streaming {
                for try await _ in session.streamResponse(to: "Weather in Paris?") {}
                for try await _ in session.streamResponse(to: "And tomorrow?") {}
            } else {
                _ = try await session.respond(to: "Weather in Paris?")
                _ = try await session.respond(to: "And tomorrow?")
            }

            // Request 2 continues the first turn after the tool ran; request 3 starts the next
            // turn from the transcript. A provider's prompt cache only matches an identical
            // prefix, so the tool call must be sent the same way both times, including the
            // assistant's text when the model wrote some alongside its calls.
            let bodies = ToolCallHistoryURLProtocol.recordedBodies
            try #require(bodies.count == 3)
            let inFlight = try toolCallMessage(in: bodies[1])
            let replayed = try toolCallMessage(in: bodies[2])
            if content.isEmpty {
                #expect(inFlight["content"] is NSNull)
            } else {
                #expect(inFlight["content"] as? String == content)
            }
            #expect(NSDictionary(dictionary: inFlight).isEqual(to: replayed))
        }

        /// The text of each assistant message without tool calls, as sent in a request body.
        private func assistantTexts(in body: Data) throws -> [String] {
            let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            let messages = try #require(json["messages"] as? [[String: Any]])
            return messages.filter { $0["role"] as? String == "assistant" && $0["tool_calls"] == nil }.map {
                if let text = $0["content"] as? String { return text }
                let blocks = $0["content"] as? [[String: Any]] ?? []
                return blocks.compactMap { $0["text"] as? String }.joined()
            }
        }

        @Test(arguments: [false, true])
        func replayedTurnSendsEachRoundsTextOnce(streaming: Bool) async throws {
            ToolCallHistoryURLProtocol.reset(
                responses: streaming
                    ? [toolCallStream(content: "Let me check."), textStream("Sunny."), textStream("Done.")]
                    : [toolCallCompletion(content: "Let me check."), completion("Sunny."), completion("Done.")]
            )
            let model = OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                session: ToolCallHistoryURLProtocol.makeSession()
            )
            let session = LanguageModelSession(model: model, tools: [ReorderedWeatherTool()])
            let first: String
            if streaming {
                first = try await session.streamResponse(to: "Weather in Paris?").collect().content
                for try await _ in session.streamResponse(to: "And tomorrow?") {}
            } else {
                first = try await session.respond(to: "Weather in Paris?").content
                _ = try await session.respond(to: "And tomorrow?")
            }

            // The caller gets the text of every round,
            // but the next turn sends the tool round's text only with its calls.
            #expect(first == "Let me check.Sunny.")
            let bodies = ToolCallHistoryURLProtocol.recordedBodies
            try #require(bodies.count == 3)
            #expect(try toolCallMessage(in: bodies[2])["content"] as? String == "Let me check.")
            #expect(try assistantTexts(in: bodies[2]) == ["Sunny."])
        }
    }

    /// Writes its arguments in the opposite order from its schema, like models often do, so
    /// the raw arguments and the decoded-and-re-encoded ones differ.
    private struct ReorderedWeatherTool: Tool {
        let name = "getWeather"
        let description = "Weather for a city"

        @Generable
        struct Arguments {
            var city: String
            var unit: String
        }

        func call(arguments: Arguments) async throws -> String { "Sunny" }
    }

    /// A `URLProtocol` with its own queue of JSON responses,
    /// so this suite doesn't share state with suites that run in parallel.
    private final class ToolCallHistoryURLProtocol: URLProtocol {
        private struct State: Sendable {
            var pending: [String] = []
            var recordedBodies: [Data] = []
        }

        private static let state = Locked(State())

        static func reset(responses: [String]) {
            state.withLock { $0 = State(pending: responses) }
        }

        static var recordedBodies: [Data] {
            state.withLock { $0.recordedBodies }
        }

        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [ToolCallHistoryURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            // URLSession moves `httpBody` to `httpBodyStream` before the protocol sees the request.
            let body = request.httpBody ?? request.httpBodyStream.map(Self.readAll) ?? Data()
            let next = Self.state.withLock { state -> String? in
                state.recordedBodies.append(body)
                return state.pending.isEmpty ? nil : state.pending.removeFirst()
            }
            guard let next, let url = request.url else {
                client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
                return
            }
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": next.hasPrefix("data:") ? "text/event-stream" : "application/json"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(next.utf8))
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}

        private static func readAll(_ stream: InputStream) -> Data {
            stream.open()
            defer { stream.close() }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while true {
                let read = stream.read(&buffer, maxLength: buffer.count)
                guard read > 0 else { break }
                data.append(buffer, count: read)
            }
            return data
        }
    }
#endif
