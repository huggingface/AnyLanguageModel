import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Provider schema responses", .serialized)
    struct ProviderSchemaResponseTests {
        enum Provider: CaseIterable {
            case chat, responses, openResponses, anthropic, gemini, ollama

            func session() -> LanguageModelSession {
                let http = SchemaURLProtocol.makeSession()
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
                return LanguageModelSession(model: model)
            }

            func response(text: String) -> [String: Any] {
                switch self {
                case .chat:
                    return ["id": "test", "choices": [["message": ["role": "assistant", "content": text]]]]
                case .responses, .openResponses:
                    return [
                        "id": "test",
                        "output": [["type": "message", "content": [["type": "output_text", "text": text]]]],
                    ]
                case .anthropic:
                    return [
                        "id": "test", "type": "message", "role": "assistant", "model": "test",
                        "content": [["type": "text", "text": text]],
                    ]
                case .gemini:
                    return ["candidates": [["content": ["role": "model", "parts": [["text": text]]]]]]
                case .ollama:
                    return [
                        "model": "test", "created_at": "2026-09-11T00:00:00.000Z",
                        "message": ["role": "assistant", "content": text], "done": true,
                    ]
                }
            }

            func stream(text: String) throws -> String {
                let events: [[String: Any]]
                switch self {
                case .chat:
                    events = [
                        ["id": "test", "choices": [["delta": ["content": text]]]],
                        ["id": "test", "choices": [["delta": [:], "finish_reason": "stop"]]],
                    ]
                case .responses, .openResponses:
                    events = [
                        ["type": "response.output_text.delta", "delta": text],
                        ["type": "response.completed", "response": response(text: text)],
                    ]
                case .anthropic:
                    events = [
                        ["type": "message_start", "message": response(text: "")],
                        ["type": "content_block_delta", "index": 0, "delta": ["type": "text_delta", "text": text]],
                        ["type": "message_stop"],
                    ]
                case .gemini:
                    events = [response(text: text)]
                case .ollama:
                    return try json(response(text: text)) + "\n"
                }
                return try events.map { "data: \(try json($0))\n\n" }.joined()
                    + (self == .chat ? "data: [DONE]\n\n" : "")
            }

            func requestSchema(_ body: [String: Any]) throws -> [String: Any] {
                let schema: Any?
                switch self {
                case .chat:
                    let format = body["response_format"] as? [String: Any]
                    let jsonSchema = format?["json_schema"] as? [String: Any]
                    #expect(jsonSchema?["strict"] as? Bool == true)
                    schema = jsonSchema?["schema"]
                case .responses, .openResponses:
                    let text = body["text"] as? [String: Any]
                    let format = text?["format"] as? [String: Any]
                    #expect(format?["strict"] as? Bool == true)
                    schema = format?["schema"]
                case .anthropic:
                    let config = body["output_config"] as? [String: Any]
                    let format = config?["format"] as? [String: Any]
                    schema = format?["schema"]
                case .gemini:
                    schema = (body["generationConfig"] as? [String: Any])?["responseSchema"]
                case .ollama:
                    schema = body["format"]
                }
                return try #require(schema as? [String: Any])
            }
        }

        @Test(arguments: Provider.allCases, [false, true])
        func callerSchemaReachesRequest(_ provider: Provider, _ streaming: Bool) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let detail = DynamicGenerationSchema(
                name: "Detail",
                properties: [.init(name: "answer", schema: .init(type: String.self))]
            )
            let schema = try GenerationSchema(
                root: DynamicGenerationSchema(
                    name: "Result",
                    properties: [
                        .init(name: "detail", schema: .init(referenceTo: "Detail")),
                        .init(name: "note", schema: .init(type: String.self), isOptional: true),
                    ]
                ),
                dependencies: [detail]
            )
            let content = #"{"detail":{"answer":"Paris"},"note":"capital"}"#
            SchemaURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: content) : json(provider.response(text: content))
            )
            let session = provider.session()
            let response =
                try await streaming
                ? session.streamResponse(to: "Capital of France?", schema: schema, includeSchemaInPrompt: false)
                    .collect()
                : session.respond(to: "Capital of France?", schema: schema, includeSchemaInPrompt: false)
            let result = try response.content.value(GeneratedContent.self, forProperty: "detail")
            #expect(try result.value(String.self, forProperty: "answer") == "Paris")
            let data = try #require(SchemaURLProtocol.recordedBodies.first)
            let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let sent = try provider.requestSchema(body)
            #expect((sent["type"] as? String)?.lowercased() == "object")
            #expect(sent["$defs"] == nil)
            let properties = try #require(sent["properties"] as? [String: [String: Any]])
            #expect(Set(properties.keys) == ["detail", "note"])
            let nested = try #require(properties["detail"])
            #expect((nested["type"] as? String)?.lowercased() == "object")
            #expect((nested["properties"] as? [String: Any])?["answer"] != nil)
            if provider == .chat || provider == .responses || provider == .openResponses {
                #expect(sent["additionalProperties"] as? Bool == false)
                #expect(nested["additionalProperties"] as? Bool == false)
                #expect(Set(try #require(sent["required"] as? [String])) == ["detail", "note"])
            }
        }
    }

    private func json(_ value: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: value), as: UTF8.self)
    }

    /// A `URLProtocol` that answers requests from a queue of canned responses and records
    /// every request body it sees, so request/response round trips can be asserted offline.
    private final class SchemaURLProtocol: URLProtocol {
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
            configuration.protocolClasses = [SchemaURLProtocol.self]
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
