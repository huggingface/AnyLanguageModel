import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Tool-calling mode for models without a tool-choice setting")
struct ToolsOfferedTests {
    @Test(arguments: [nil, .allowed, .required] as [GenerationOptions.ToolCallingMode?])
    func toolsAreOffered(mode: GenerationOptions.ToolCallingMode?) {
        let options = GenerationOptions(toolCallingMode: mode)
        #expect(options.toolsOffered(from: [WeatherTool()]).map(\.name) == ["getWeather"])
    }

    @Test func disallowedOffersNoTools() {
        let options = GenerationOptions(toolCallingMode: .disallowed)
        #expect(options.toolsOffered(from: [WeatherTool()]).isEmpty)
    }
}

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Tool-calling mode in provider requests", .serialized)
    struct ToolCallingModeRequestTests {
        typealias Kind = GenerationOptions.ToolCallingMode.Kind

        enum Provider: String, CaseIterable, Sendable, CustomTestStringConvertible {
            case chat, responses, openResponses, anthropic, gemini, ollama

            var testDescription: String { rawValue }

            func makeModel() -> any LanguageModel {
                let http = ToolCallingModeURLProtocol.makeSession()
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
                return model
            }

            func makeSession(tools: [any Tool] = [WeatherTool()]) -> LanguageModelSession {
                LanguageModelSession(model: makeModel(), tools: tools)
            }

            /// A response with text, or with a call to `getWeather` for Paris.
            func response(tool: Bool = false) -> String {
                switch self {
                case .chat:
                    let message =
                        tool
                        ? #"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Paris\"}"}}]}"#
                        : #"{"role":"assistant","content":"Hello"}"#
                    return #"{"id":"test","choices":[{"message":\#(message)}]}"#
                case .responses, .openResponses:
                    let output =
                        tool
                        ? #"{"type":"function_call","call_id":"call_1","name":"getWeather","arguments":"{\"city\":\"Paris\"}"}"#
                        : #"{"type":"message","content":[{"type":"output_text","text":"Hello"}]}"#
                    return #"{"id":"test","output":[\#(output)]}"#
                case .anthropic:
                    let content =
                        tool
                        ? #"{"type":"tool_use","id":"call_1","name":"getWeather","input":{"city":"Paris"}}"#
                        : #"{"type":"text","text":"Hello"}"#
                    return #"{"id":"test","type":"message","role":"assistant","model":"test","content":[\#(content)]}"#
                case .gemini:
                    let part =
                        tool
                        ? #"{"functionCall":{"name":"getWeather","args":{"city":"Paris"}}}"#
                        : #"{"text":"Hello"}"#
                    return #"{"candidates":[{"content":{"role":"model","parts":[\#(part)]}}]}"#
                case .ollama:
                    let message =
                        tool
                        ? #"{"role":"assistant","content":"","tool_calls":[{"function":{"name":"getWeather","arguments":{"city":"Paris"}}}]}"#
                        : #"{"role":"assistant","content":"Hello"}"#
                    let created = "2026-09-11T00:00:00.000Z"
                    return #"{"model":"test","created_at":"\#(created)","message":\#(message),"done":true}"#
                }
            }

            /// The tool choice in a request body, in the provider's format.
            func toolChoice(in body: [String: JSONValue]) -> JSONValue? {
                switch self {
                case .gemini: body["toolConfig"]?.objectValue?["functionCallingConfig"]
                default: body["tool_choice"]
                }
            }

            /// The tool choice a request should carry for a tool-calling mode.
            func expectedToolChoice(_ kind: Kind?) -> JSONValue? {
                guard let kind else { return nil }
                switch self {
                case .chat, .responses, .openResponses:
                    let values: [Kind: String] = [.allowed: "auto", .required: "required", .disallowed: "none"]
                    return .string(values[kind]!)
                case .anthropic:
                    let values: [Kind: String] = [.allowed: "auto", .required: "any", .disallowed: "none"]
                    return .object(["type": .string(values[kind]!)])
                case .gemini:
                    let values: [Kind: String] = [.allowed: "AUTO", .required: "ANY", .disallowed: "NONE"]
                    return .object(["mode": .string(values[kind]!)])
                case .ollama:
                    // Ollama has no tool-choice setting.
                    return nil
                }
            }
        }

        static let modes: [GenerationOptions.ToolCallingMode?] = [nil, .allowed, .disallowed, .required]

        private func requestBodies() throws -> [[String: JSONValue]] {
            try ToolCallingModeURLProtocol.recordedBodies.map {
                try JSONDecoder().decode([String: JSONValue].self, from: $0)
            }
        }

        @Test(arguments: Provider.allCases, modes)
        func requestCarriesMode(_ provider: Provider, mode: GenerationOptions.ToolCallingMode?) async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(provider.response())
            let options = GenerationOptions(toolCallingMode: mode)
            let response = try await provider.makeSession().respond(to: "Weather?", options: options)
            #expect(response.content == "Hello")

            let body = try #require(requestBodies().first)
            #expect(provider.toolChoice(in: body) == provider.expectedToolChoice(mode?.kind))
            if mode == nil {
                // Without a mode, the request has no tool-choice fields.
                #expect(body["tool_choice"] == nil)
                #expect(body["toolConfig"] == nil)
            }
            // Ollama honors `disallowed` by leaving the tools out.
            let offersTools = !(provider == .ollama && mode == .disallowed)
            #expect((body["tools"] != nil) == offersTools)
        }

        @Test(arguments: Provider.allCases, modes)
        func requestWithoutToolsOmitsMode(_ provider: Provider, mode: GenerationOptions.ToolCallingMode?) async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(provider.response())
            let options = GenerationOptions(toolCallingMode: mode)
            _ = try await provider.makeSession(tools: []).respond(to: "Hi", options: options)

            let body = try #require(requestBodies().first)
            #expect(body["tools"] == nil)
            #expect(body["tool_choice"] == nil)
            #expect(body["toolConfig"] == nil)
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .gemini])
        func followUpRequestKeepsRequiredMode(_ provider: Provider) async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(provider.response(tool: true))
            ToolCallingModeURLProtocol.enqueue(provider.response())
            let options = GenerationOptions(toolCallingMode: .required)
            let response = try await provider.makeSession().respond(to: "Weather?", options: options)
            #expect(response.content == "Hello")

            let bodies = try requestBodies()
            #expect(bodies.count == 2)
            for body in bodies {
                #expect(provider.toolChoice(in: body) == provider.expectedToolChoice(.required))
            }
        }

        @Test func anthropicFollowUpRequestKeepsRequiredMode() async throws {
            ToolCallingModeURLProtocol.reset()
            let message = #"{"id":"test","type":"message","role":"assistant","model":"test","content":[]}"#
            let start = #"{"type":"message_start","message":\#(message)}"#
            ToolCallingModeURLProtocol.enqueue(
                eventStream: [
                    start,
                    #"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"getWeather","input":{}}}"#,
                    #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}"#,
                    #"{"type":"content_block_stop","index":0}"#,
                    #"{"type":"message_stop"}"#,
                ]
            )
            ToolCallingModeURLProtocol.enqueue(
                eventStream: [
                    start,
                    #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
                    #"{"type":"message_stop"}"#,
                ]
            )
            let options = GenerationOptions(toolCallingMode: .required)
            let response = try await Provider.anthropic.makeSession()
                .streamResponse(to: "Weather?", options: options)
                .collect()
            #expect(response.content == "Hello")

            let bodies = try requestBodies()
            #expect(bodies.count == 2)
            for body in bodies {
                #expect(body["tool_choice"] == .object(["type": .string("any")]))
            }
        }

        @Test func requiredModeEndsAtToolRoundLimit() async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(Provider.chat.response(tool: true))
            ToolCallingModeURLProtocol.enqueue(Provider.chat.response(tool: true))
            var options = GenerationOptions(toolCallingMode: .required)
            options.maximumToolRounds = 1

            await #expect(throws: LanguageModelSession.ToolRoundLimitExceeded.self) {
                try await Provider.chat.makeSession().respond(to: "Weather?", options: options)
            }
            let bodies = try requestBodies()
            #expect(bodies.count == 2)
            for body in bodies {
                #expect(body["tool_choice"] == .string("required"))
            }
        }

        @Test func requiredModeEndsWhenDynamicInstructionsDropTools() async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(Provider.chat.response(tool: true))
            ToolCallingModeURLProtocol.enqueue(Provider.chat.response())
            let session = LanguageModelSession(
                model: Provider.chat.makeModel(),
                dynamicInstructions: WeatherUntilCalled(called: Locked(false))
            )
            let options = GenerationOptions(toolCallingMode: .required)
            let response = try await session.respond(to: "Weather?", options: options)
            #expect(response.content == "Hello")

            let bodies = try requestBodies()
            #expect(bodies.count == 2)
            #expect(bodies.first?["tool_choice"] == .string("required"))
            // The follow-up request has no tools, so it has no tool choice.
            #expect(bodies.last?["tools"] == nil)
            #expect(bodies.last?["tool_choice"] == nil)
        }

        @Test func anthropicCustomToolChoiceOverridesMode() async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(Provider.anthropic.response())
            var options = GenerationOptions(toolCallingMode: .disallowed)
            options[custom: AnthropicLanguageModel.self] = .init(toolChoice: .tool(name: "getWeather"))
            _ = try await Provider.anthropic.makeSession().respond(to: "Weather?", options: options)

            let body = try #require(requestBodies().first)
            #expect(body["tool_choice"] == .object(["type": .string("tool"), "name": .string("getWeather")]))
        }

        @Test func openResponsesCustomToolChoiceOverridesMode() async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(Provider.openResponses.response())
            var options = GenerationOptions(toolCallingMode: .required)
            options[custom: OpenResponsesLanguageModel.self] = .init(toolChoice: .auto)
            _ = try await Provider.openResponses.makeSession().respond(to: "Weather?", options: options)

            let body = try #require(requestBodies().first)
            #expect(body["tool_choice"] == .string("auto"))
        }

        @Test(arguments: [Provider.chat, .responses, .openResponses, .anthropic])
        func extraBodyOverridesMode(_ provider: Provider) async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(provider.response())
            var options = GenerationOptions(toolCallingMode: .required)
            let extraBody: [String: JSONValue] = ["tool_choice": .string("none")]
            switch provider {
            case .chat, .responses:
                options[custom: OpenAILanguageModel.self] = .init(extraBody: extraBody)
            case .openResponses:
                options[custom: OpenResponsesLanguageModel.self] = .init(extraBody: extraBody)
            default:
                options[custom: AnthropicLanguageModel.self] = .init(extraBody: extraBody)
            }
            _ = try await provider.makeSession().respond(to: "Weather?", options: options)

            let body = try #require(requestBodies().first)
            #expect(body["tool_choice"] == .string("none"))
        }

        @Test func geminiModeJoinsRetrievalConfig() async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(Provider.gemini.response())
            var options = GenerationOptions(toolCallingMode: .required)
            options[custom: GeminiLanguageModel.self] = .init(
                serverTools: [.googleMaps(latitude: 37.33, longitude: -122.01)]
            )
            _ = try await Provider.gemini.makeSession().respond(to: "Weather?", options: options)

            let body = try #require(requestBodies().first)
            let toolConfig = try #require(body["toolConfig"]?.objectValue)
            #expect(toolConfig["functionCallingConfig"] == .object(["mode": .string("ANY")]))
            #expect(toolConfig["retrievalConfig"] != nil)
        }

        @Test func geminiServerToolsAloneOmitMode() async throws {
            ToolCallingModeURLProtocol.reset()
            ToolCallingModeURLProtocol.enqueue(Provider.gemini.response())
            var options = GenerationOptions(toolCallingMode: .required)
            options[custom: GeminiLanguageModel.self] = .init(serverTools: [.googleSearch])
            _ = try await Provider.gemini.makeSession(tools: []).respond(to: "Weather?", options: options)

            let body = try #require(requestBodies().first)
            #expect(body["tools"] != nil)
            #expect(body["toolConfig"] == nil)
        }
    }

    /// Offers `getWeather` until it's called,
    /// like the dynamic profile in the Foundation Models `ToolCallingMode` example.
    private struct WeatherUntilCalled: DynamicInstructions {
        let called: Locked<Bool>

        var body: some DynamicInstructions {
            Instructions("Answer questions about the weather.")
            if !called.withLock({ $0 }) {
                [OneShotWeatherTool(called: called)] as [any Tool]
            }
        }
    }

    private struct OneShotWeatherTool: Tool {
        let name = "getWeather"
        let description = "Retrieve the latest weather information for a city"
        let called: Locked<Bool>

        func call(arguments: WeatherTool.Arguments) async throws -> String {
            called.withLock { $0 = true }
            return try await WeatherTool().call(arguments: arguments)
        }
    }

    /// Answers requests from a queue of canned responses and records each request body.
    /// Its state is separate from other suites' fixtures, so suites can run concurrently.
    private final class ToolCallingModeURLProtocol: URLProtocol {
        private struct State: Sendable {
            var pending: [Data] = []
            var recordedBodies: [Data] = []
        }

        private static let state = Locked(State())

        static func reset() {
            state.withLock { $0 = State() }
        }

        static func enqueue(_ json: String) {
            state.withLock { $0.pending.append(Data(json.utf8)) }
        }

        static func enqueue(eventStream events: [String]) {
            let body = events.map { "data: \($0)\n\n" }.joined()
            state.withLock { $0.pending.append(Data(body.utf8)) }
        }

        static var recordedBodies: [Data] {
            state.withLock { $0.recordedBodies }
        }

        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [ToolCallingModeURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            // URLSession moves `httpBody` to `httpBodyStream` before the protocol sees the request.
            let body = request.httpBody ?? request.httpBodyStream.map(Self.readAll) ?? Data()
            let pending = Self.state.withLock { state -> Data? in
                state.recordedBodies.append(body)
                return state.pending.isEmpty ? nil : state.pending.removeFirst()
            }
            guard let pending, let url = request.url else {
                client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
                return
            }
            let streaming = request.value(forHTTPHeaderField: "Accept") == "text/event-stream"
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": streaming ? "text/event-stream" : "application/json"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: pending)
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
                if read <= 0 { break }
                data.append(buffer, count: read)
            }
            return data
        }
    }
#endif
