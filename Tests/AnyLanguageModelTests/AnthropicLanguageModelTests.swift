import Foundation
import Testing

@testable import AnyLanguageModel

private let anthropicAPIKey: String? = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]

@Generable
private struct AnthropicStructuredForecast {
    var summary: String
    var temperatureCelsius: Int
}

@Suite("AnthropicLanguageModel", .serialized, .enabled(if: anthropicAPIKey?.isEmpty == false))
struct AnthropicLanguageModelTests {
    let model = AnthropicLanguageModel(
        apiKey: anthropicAPIKey!,
        model: "claude-sonnet-4-5"
    )

    @Test func customHost() throws {
        let customURL = URL(string: "https://example.com")!
        let model = AnthropicLanguageModel(baseURL: customURL, apiKey: "test", model: "test-model")
        #expect(model.baseURL.absoluteString.hasSuffix("/"))
    }

    @Test func basicResponse() async throws {
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: "Say hello")
        #expect(!response.content.isEmpty)
    }

    @Test func withInstructions() async throws {
        let session = LanguageModelSession(
            model: model,
            instructions: "You are a helpful assistant. Be concise."
        )

        let response = try await session.respond(to: "What is 2+2?")
        #expect(!response.content.isEmpty)
    }

    @Test func streaming() async throws {
        let session = LanguageModelSession(model: model)

        let stream = session.streamResponse(to: "Count to 5")
        var chunks: [String] = []

        for try await response in stream {
            chunks.append(response.content)
        }

        #expect(!chunks.isEmpty)
    }

    @Test func streamingString() async throws {
        let session = LanguageModelSession(model: model)

        let stream = session.streamResponse(to: "Say 'Hello' slowly")

        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
        }

        #expect(!snapshots.isEmpty)
        #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
    }

    @Test func streamingStructured() async throws {
        let session = LanguageModelSession(model: model)

        let stream = session.streamResponse(
            to: "Provide a short weather forecast summary and a celsius temperature.",
            generating: AnthropicStructuredForecast.self
        )

        var snapshots: [LanguageModelSession.ResponseStream<AnthropicStructuredForecast>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
        }

        #expect(!snapshots.isEmpty)
        #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        #expect(!(snapshots.last!.content.summary ?? "").isEmpty)
    }

    @Test func withGenerationOptions() async throws {
        let session = LanguageModelSession(model: model)

        let options = GenerationOptions(
            temperature: 0.7,
            maximumResponseTokens: 50
        )

        let response = try await session.respond(
            to: "Tell me a fact",
            options: options
        )
        #expect(!response.content.isEmpty)
    }

    @Test func structuredResponse() async throws {
        let session = LanguageModelSession(model: model)

        let response = try await session.respond(
            to: "Summarize the weather with a short summary and a celsius temperature.",
            generating: AnthropicStructuredForecast.self
        )

        #expect(!response.content.summary.isEmpty)
        #expect(response.rawContent.jsonString.contains("summary"))
    }

    @Test func conversationContext() async throws {
        let session = LanguageModelSession(model: model)

        let firstResponse = try await session.respond(to: "My favorite color is blue")
        #expect(!firstResponse.content.isEmpty)

        let secondResponse = try await session.respond(to: "What did I just tell you?")
        #expect(secondResponse.content.contains("color"))
    }

    @Test func withTools() async throws {
        let weatherTool = WeatherTool()
        let session = LanguageModelSession(model: model, tools: [weatherTool])

        let response = try await session.respond(to: "How's the weather in San Francisco?")

        var foundToolOutput = false
        for case let .toolOutput(toolOutput) in response.transcriptEntries {
            #expect(!toolOutput.id.isEmpty)
            #expect(toolOutput.toolName == "getWeather")
            foundToolOutput = true
        }
        #expect(foundToolOutput)
    }

    @Test func multimodalWithImageURL() async throws {
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(
            to: "Describe this image",
            image: .init(url: testImageURL)
        )
        #expect(!response.content.isEmpty)
    }

    @Test func multimodalWithImageData() async throws {
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(
            to: "Describe this image",
            image: .init(data: testImageData, mimeType: "image/png")
        )
        #expect(!response.content.isEmpty)
    }
}

#if canImport(Darwin) && !canImport(AsyncHTTPClient)

    @Generable
    private struct AnthropicRequestAnswer {
        var answer: String
    }

    @Suite("Anthropic request encoding", .serialized)
    struct AnthropicRequestTests {
        private func makeSession() -> LanguageModelSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [AnthropicRequestURLProtocol.self]
            let model = AnthropicLanguageModel(
                apiKey: "test-key",
                model: "test-model",
                session: URLSession(configuration: configuration)
            )
            return LanguageModelSession(model: model)
        }

        private func requestBody() throws -> [String: JSONValue] {
            let body = try #require(AnthropicRequestURLProtocol.body.withLock { $0 })
            return try JSONDecoder().decode([String: JSONValue].self, from: body)
        }

        @Test(arguments: [false, true], [false, true])
        func invalidThinkingFailsBeforeSending(adaptive: Bool, streaming: Bool) async throws {
            AnthropicRequestURLProtocol.body.withLock { $0 = nil }
            var thinking = AnthropicLanguageModel.CustomGenerationOptions.Thinking.enabled(budgetTokens: 2048)
            if adaptive {
                thinking.type = .adaptive
            } else {
                thinking.budgetTokens = nil
            }
            var options = GenerationOptions(maximumResponseTokens: 4096)
            options[custom: AnthropicLanguageModel.self] = .init(thinking: thinking)
            let session = makeSession()

            do {
                if streaming {
                    for try await _ in session.streamResponse(to: "Answer", options: options) {}
                } else {
                    _ = try await session.respond(to: "Answer", options: options)
                }
                Issue.record("Expected invalid thinking configuration to fail before sending")
            } catch EncodingError.invalidValue(_, let context) {
                #expect(context.debugDescription.contains("token budget"))
            }
            #expect(AnthropicRequestURLProtocol.body.withLock { $0 } == nil)
        }

        @Test(arguments: [
            AnthropicLanguageModel.CustomGenerationOptions.Effort.low, .medium, .high, .extraHigh, .max,
        ])
        func effortPreservesStructuredOutput(effort: AnthropicLanguageModel.CustomGenerationOptions.Effort) async throws
        {
            var options = GenerationOptions(maximumResponseTokens: 4096)
            options[custom: AnthropicLanguageModel.self] = .init(
                thinking: .adaptive(display: .omitted),
                serviceTier: .standard,
                effort: effort
            )
            let response = try await makeSession().respond(
                to: "Answer",
                generating: AnthropicRequestAnswer.self,
                options: options
            )
            #expect(response.content.answer == "Hello")
            let body = try requestBody()
            #expect(body["thinking"] == .object(["type": .string("adaptive"), "display": .string("omitted")]))
            #expect(body["service_tier"] == .string("standard"))
            #expect(body["max_tokens"] == .int(4096))
            #expect(body["effort"] == nil)
            let outputConfig = try #require(body["output_config"]?.objectValue)
            #expect(outputConfig["effort"] == .string(effort == .extraHigh ? "xhigh" : effort.rawValue))
            let format = try #require(outputConfig["format"]?.objectValue)
            #expect(format["type"] == .string("json_schema"))
            #expect(format["schema"]?.objectValue?["properties"]?.objectValue?["answer"] != nil)
        }

        @Test func enabledThinkingAndEffortWithoutSchema() async throws {
            var options = GenerationOptions(maximumResponseTokens: 4096)
            options[custom: AnthropicLanguageModel.self] = .init(
                thinking: .enabled(budgetTokens: 2048, display: .summarized),
                effort: .medium
            )
            let response = try await makeSession().respond(to: "Answer", options: options)
            #expect(response.content == "Hello")
            let body = try requestBody()
            #expect(
                body["thinking"]
                    == .object([
                        "type": .string("enabled"), "budget_tokens": .int(2048), "display": .string("summarized"),
                    ])
            )
            #expect(body["output_config"] == .object(["effort": .string("medium")]))
            #expect(body["stream"] == nil)
        }

        @Test func defaultsOmitThinkingAndEffort() async throws {
            _ = try await makeSession().respond(to: "Answer")
            let body = try requestBody()
            #expect(body["thinking"] == nil)
            #expect(body["output_config"] == nil)
        }

        @Test func extraBodyOverridesCustomOptions() async throws {
            var options = GenerationOptions()
            options[custom: AnthropicLanguageModel.self] = .init(
                thinking: .adaptive(),
                extraBody: ["output_config": .object(["effort": .string("low")])],
                effort: .high
            )
            _ = try await makeSession().respond(to: "Answer", options: options)
            let body = try requestBody()
            #expect(body["thinking"] == .object(["type": .string("adaptive")]))
            #expect(body["output_config"] == .object(["effort": .string("low")]))
        }

        @Test(arguments: [
            AnthropicLanguageModel.CustomGenerationOptions.Thinking.ThinkingDisplay.omitted, .summarized,
        ])
        func streamingIgnoresThinkingAndSignatureDeltas(
            display: AnthropicLanguageModel.CustomGenerationOptions.Thinking.ThinkingDisplay
        ) async throws {
            var options = GenerationOptions()
            options[custom: AnthropicLanguageModel.self] = .init(thinking: .adaptive(display: display), effort: .low)
            var text = ""
            for try await snapshot in makeSession().streamResponse(to: "Answer", options: options) {
                text = snapshot.content
            }
            #expect(text == "Hello")
            let body = try requestBody()
            #expect(body["stream"] == .bool(true))
            #expect(body["thinking"] == .object(["type": .string("adaptive"), "display": .string(display.rawValue)]))
            #expect(body["output_config"] == .object(["effort": .string("low")]))
        }
    }

    /// Separate state from other providers' fixtures so suites can run concurrently.
    private final class AnthropicRequestURLProtocol: URLProtocol {
        static let body = Locked<Data?>(nil)

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func stopLoading() {}

        override func startLoading() {
            do {
                var data = request.httpBody ?? Data()
                if let stream = request.httpBodyStream {
                    stream.open()
                    defer { stream.close() }
                    var buffer = [UInt8](repeating: 0, count: 4096)
                    while true {
                        let count = stream.read(&buffer, maxLength: buffer.count)
                        if count < 0 { throw stream.streamError ?? URLError(.cannotDecodeContentData) }
                        if count == 0 { break }
                        data.append(buffer, count: count)
                    }
                }
                Self.body.withLock { $0 = data }
                let json = try JSONDecoder().decode([String: JSONValue].self, from: data)
                let streaming = json["stream"] == .bool(true)
                let responseText: String
                if streaming {
                    let thinkingDelta =
                        json["thinking"]?.objectValue?["display"] == .string("summarized")
                        ? """
                        event: content_block_delta
                        data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Considering"}}


                        """ : ""
                    responseText =
                        thinkingDelta + """
                            event: content_block_delta
                            data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"test-signature"}}

                            event: content_block_delta
                            data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}

                            event: message_stop
                            data: {"type":"message_stop"}


                            """
                } else {
                    let structured = json["output_config"]?.objectValue?["format"] != nil
                    let text = structured ? #"{\"answer\":\"Hello\"}"# : "Hello"
                    responseText = """
                        {"id":"test","type":"message","role":"assistant","model":"test-model","stop_reason":"end_turn",
                         "content":[{"type":"thinking","thinking":"","signature":"test-signature"},{"type":"text","text":"\(text)"}]}
                        """
                }
                let response = HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: ["Content-Type": streaming ? "text/event-stream" : "application/json"]
                )!
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: Data(responseText.utf8))
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }
    }

#endif
