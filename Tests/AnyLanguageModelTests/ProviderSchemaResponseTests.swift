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

        @Test(arguments: [false, true])
        func ollamaRequestBytesAreStable(_ streaming: Bool) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let content = #"{"alpha":{"count":1,"name":"a"},"beta":{"enabled":true,"score":2}}"#
            let optionEntries: [(String, JSONValue)] = [
                ("seed", 42), ("temperature", 0.5), ("stop", ["END", "STOP"]),
                ("think", false), ("keep_alive", .null),
            ]
            for (index, schema) in try DynamicGenerationSchemaTests.schemasInDifferentInsertionOrders().enumerated() {
                var customOptions: [String: JSONValue] = [:]
                for (key, value) in index == 0 ? optionEntries : Array(optionEntries.reversed()) {
                    customOptions[key] = value
                }
                var options = GenerationOptions()
                options[custom: OllamaLanguageModel.self] = customOptions
                SchemaURLProtocol.enqueue(
                    json: try streaming
                        ? Provider.ollama.stream(text: content) : json(Provider.ollama.response(text: content))
                )
                let session = Provider.ollama.session()
                if streaming {
                    _ = try await session.streamResponse(
                        to: "Return a result",
                        schema: schema,
                        includeSchemaInPrompt: false,
                        options: options
                    )
                    .collect()
                } else {
                    _ = try await session.respond(
                        to: "Return a result",
                        schema: schema,
                        includeSchemaInPrompt: false,
                        options: options
                    )
                }
            }
            let bodies = SchemaURLProtocol.recordedBodies
            #expect(bodies.count == 2)
            let first = try #require(bodies.first)
            #expect(bodies.last == first)

            // Check the actual request encoder's sorted-key setting, including nested objects.
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            for body in bodies {
                let params = try JSONDecoder().decode([String: JSONValue].self, from: body)
                #expect(try encoder.encode(params) == body)
                #expect(params["stream"] == .bool(streaming))
                #expect(params["think"] == .bool(false))
                #expect(params["keep_alive"] == .null)
                #expect(params["options"] == .object(["seed": 42, "temperature": 0.5, "stop": ["END", "STOP"]]))
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

    extension ProviderSchemaResponseTests {
        /// A constant string reaches Anthropic as `const`, and the other providers,
        /// whose schemas don't support `const`, as a one-choice `enum`.
        @Test(arguments: Provider.allCases)
        func constantStringReachesRequest(_ provider: Provider) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let content = #"{"kind":"fixed","described":"fixed","choice":"a"}"#
            SchemaURLProtocol.enqueue(json: try json(provider.response(text: content)))
            let response = try await provider.session().respond(
                to: "Classify",
                generating: StringChoiceGuided.self,
                includeSchemaInPrompt: false
            )
            #expect(response.content.kind == "fixed")

            let data = try #require(SchemaURLProtocol.recordedBodies.first)
            let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let properties = try #require(provider.requestSchema(body)["properties"] as? [String: [String: Any]])
            let kind = try #require(properties["kind"])
            if provider == .anthropic {
                #expect(kind["const"] as? String == "fixed")
                #expect(kind["enum"] == nil)
            } else {
                #expect(kind["enum"] as? [String] == ["fixed"])
                #expect(kind["const"] == nil)
            }
            #expect(properties["choice"]?["enum"] as? [String] == ["a", "b"])
        }
    }

    extension ProviderSchemaResponseTests {
        /// The text of the response that the session's transcript records last.
        private func lastResponseText(in session: LanguageModelSession) -> String? {
            guard case .response(let response)? = session.transcript.last,
                case .text(let text)? = response.segments.first
            else { return nil }
            return text.content
        }

        /// The JSON value in a string, which compares without regard to the order of keys.
        private func jsonValue(_ text: String?) throws -> JSONValue? {
            try text.map { try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8)) }
        }

        /// An enumeration whose cases have associated values has a union at the root of its schema.
        /// OpenAI and Open Responses need an object root, and Gemini's response schemas have no unions,
        /// so they get the union as the `response` property of an object,
        /// and the content they return is unwrapped.
        /// Anthropic and Ollama get the union itself.
        @Test(arguments: Provider.allCases, [false, true])
        func unionRootReachesRequest(_ provider: Provider, _ streaming: Bool) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let wraps = provider != .anthropic && provider != .ollama
            let operand = #"{"type":"number","value":1}"#
            let text = wraps ? #"{"response":\#(operand)}"# : operand
            SchemaURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: text) : json(provider.response(text: text))
            )
            let session = provider.session()
            let response =
                try await streaming
                ? session.streamResponse(to: "One", generating: Operand.self, includeSchemaInPrompt: false).collect()
                : session.respond(to: "One", generating: Operand.self, includeSchemaInPrompt: false)
            #expect(response.content == .number(1))
            #expect(try jsonValue(response.rawContent.jsonString) == jsonValue(operand))
            #expect(try jsonValue(lastResponseText(in: session)) == jsonValue(operand))

            let data = try #require(SchemaURLProtocol.recordedBodies.first)
            let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let sent = try provider.requestSchema(body)
            #expect(sent["$defs"] == nil)
            var union = sent
            if wraps {
                #expect((sent["type"] as? String)?.lowercased() == "object")
                #expect(sent["required"] as? [String] == ["response"])
                let properties = try #require(sent["properties"] as? [String: [String: Any]])
                #expect(Set(properties.keys) == ["response"])
                union = try #require(properties["response"])
                if provider != .gemini {
                    #expect(sent["additionalProperties"] as? Bool == false)
                }
            }
            let cases = try #require(union["anyOf"] as? [[String: Any]])
            let types = cases.map { ($0["properties"] as? [String: [String: Any]])?["type"] }
            #expect(types.count == 2)
            for type in types {
                #expect(type?["const"] != nil || type?["enum"] != nil)
            }
        }

        /// OpenAI and Open Responses need an object root, so an array is wrapped too.
        /// Gemini, Anthropic, and Ollama accept an array at the root.
        @Test(arguments: Provider.allCases, [false, true])
        func arrayRootIsWrappedOnlyForObjectRootProviders(_ provider: Provider, _ streaming: Bool) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let wraps = provider == .chat || provider == .responses || provider == .openResponses
            let array = #"["a","b"]"#
            let text = wraps ? #"{"response":\#(array)}"# : array
            SchemaURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: text) : json(provider.response(text: text))
            )
            let session = provider.session()
            let response =
                try await streaming
                ? session.streamResponse(to: "Two", generating: [String].self, includeSchemaInPrompt: false).collect()
                : session.respond(to: "Two", generating: [String].self, includeSchemaInPrompt: false)
            #expect(response.content == ["a", "b"])
            #expect(try jsonValue(lastResponseText(in: session)) == jsonValue(array))

            let data = try #require(SchemaURLProtocol.recordedBodies.first)
            let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let sent = try provider.requestSchema(body)
            if wraps {
                #expect(sent["type"] as? String == "object")
                let properties = try #require(sent["properties"] as? [String: [String: Any]])
                #expect(properties["response"]?["type"] as? String == "array")
            } else {
                #expect((sent["type"] as? String)?.lowercased() == "array")
            }
        }

        /// A provider that doesn't apply the response schema can return content without the wrapper,
        /// and an object with properties besides the wrapper property isn't unwrapped.
        @Test(arguments: [false, true])
        func unwrappedResponseIsUsedAsSent(_ streaming: Bool) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let provider = Provider.chat
            let operand = #"{"type":"number","value":1}"#
            SchemaURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: operand) : json(provider.response(text: operand))
            )
            let session = provider.session()
            let response =
                try await streaming
                ? session.streamResponse(to: "One", generating: Operand.self).collect()
                : session.respond(to: "One", generating: Operand.self)
            #expect(response.content == .number(1))

            let extra = try GeneratedContent(json: #"{"response":{"type":"number","value":1},"note":"extra"}"#)
            #expect(ProviderResponseSchema.unwrapping(extra, fromProperty: "response") == extra)
        }
    }

    extension ProviderSchemaResponseTests {
        @Generable struct Address: Equatable { var street: String; var unit: String? }
        @Generable struct Contact { var name: String; var address: Address; var previous: [Address]; var step: Step }

        /// OpenAI's strict mode requires every property of every object in `required`,
        /// so an optional property of a nested object, of an array's objects,
        /// or of an enumeration case is required and allows `null`.
        @Test(arguments: [Provider.chat, .responses, .openResponses])
        func strictModeRequiresNestedOptionalProperties(_ provider: Provider) async throws {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            let content = """
                {"name":"Ada","address":{"street":"Main","unit":null},\
                "previous":[{"street":"Old","unit":"2"}],"step":{"type":"note","value":null}}
                """
            SchemaURLProtocol.enqueue(json: try json(provider.response(text: content)))
            let response = try await provider.session().respond(
                to: "Contact",
                generating: Contact.self,
                includeSchemaInPrompt: false
            )
            #expect(response.content.address == Address(street: "Main", unit: nil))
            #expect(response.content.previous == [Address(street: "Old", unit: "2")])
            #expect(response.content.step == .note(nil))

            let data = try #require(SchemaURLProtocol.recordedBodies.first)
            let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let sent = try provider.requestSchema(body)
            #expect(sent["required"] as? [String] == ["address", "name", "previous", "step"])
            let properties = try #require(sent["properties"] as? [String: [String: Any]])
            let previous = try #require(properties["previous"]?["items"] as? [String: Any])
            for address in [try #require(properties["address"]), previous] {
                #expect(address["required"] as? [String] == ["street", "unit"])
                #expect(address["additionalProperties"] as? Bool == false)
                let unit = (address["properties"] as? [String: [String: Any]])?["unit"]
                #expect(unit?["type"] as? [String] == ["string", "null"])
            }
            let cases = try #require(properties["step"]?["anyOf"] as? [[String: Any]])
            let note = try #require(
                cases.first {
                    let properties = $0["properties"] as? [String: [String: Any]]
                    return properties?["type"]?["enum"] as? [String] == ["note"]
                }
            )
            #expect(note["required"] as? [String] == ["type", "value"])
            #expect(note["additionalProperties"] as? Bool == false)
            let value = (note["properties"] as? [String: [String: Any]])?["value"]
            #expect(value?["type"] as? [String] == ["string", "null"])
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

    @Generable
    private struct SchemaPromptGreeting {
        let text: String
    }

    /// The text of the user message that says `prompt`, in whichever shape the provider sends it.
    func userMessage(saying prompt: String, in body: Any) -> String? {
        if let message = body as? [String: Any], message["role"] as? String == "user",
            let data = try? JSONSerialization.data(withJSONObject: message),
            let text = String(data: data, encoding: .utf8), text.contains(prompt)
        {
            return text
        }
        let children: [Any] =
            if let dictionary = body as? [String: Any] { Array(dictionary.values) } else { body as? [Any] ?? [] }
        return children.lazy.compactMap { userMessage(saying: prompt, in: $0) }.first
    }

    extension ProviderSchemaResponseTests {
        /// The body of the request the session sent, after answering `content`.
        private func sentBody(
            _ provider: Provider,
            streaming: Bool,
            content: String,
            ask: (LanguageModelSession) async throws -> Void
        ) async throws -> [String: Any] {
            SchemaURLProtocol.reset()
            defer { SchemaURLProtocol.reset() }
            SchemaURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: content) : json(provider.response(text: content))
            )
            try await ask(provider.session())
            let data = try #require(SchemaURLProtocol.recordedBodies.first)
            return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        }

        /// A model whose server ignores the schema it's sent only knows what to write from the prompt.
        @Test(arguments: Provider.allCases, [false, true])
        func aStructuredRequestSaysItsSchema(_ provider: Provider, _ streaming: Bool) async throws {
            let body = try await sentBody(provider, streaming: streaming, content: #"{"text":"Hi"}"#) { session in
                let reply =
                    try await streaming
                    ? session.streamResponse(to: "Hello", generating: SchemaPromptGreeting.self).collect().content
                    : session.respond(to: "Hello", generating: SchemaPromptGreeting.self).content
                #expect(reply.text == "Hi")
            }
            // Said after what's asked, where a model still reads it after rounds of tool calls.
            let message = try #require(userMessage(saying: "Hello", in: body))
            let schema = try #require(message.range(of: "Respond with valid JSON matching this object schema"))
            #expect(message.range(of: "Hello")!.upperBound <= schema.lowerBound)
            #expect(message.contains("text"))
            // The server is still asked for the schema, for those that enforce it.
            _ = try provider.requestSchema(body)
        }

        @Test(arguments: Provider.allCases)
        func aStructuredRequestLeavesItsSchemaOutWhenAskedTo(_ provider: Provider) async throws {
            let body = try await sentBody(provider, streaming: false, content: #"{"text":"Hi"}"#) { session in
                _ = try await session.respond(
                    to: "Hello",
                    generating: SchemaPromptGreeting.self,
                    includeSchemaInPrompt: false
                )
            }
            #expect(userMessage(saying: "Hello", in: body)?.contains("Respond with valid JSON") == false)
        }

        @Test(arguments: Provider.allCases)
        func aTextRequestHasNoSchema(_ provider: Provider) async throws {
            let body = try await sentBody(provider, streaming: false, content: "Hi") { session in
                _ = try await session.respond(to: "Hello")
            }
            #expect(userMessage(saying: "Hello", in: body)?.contains("Respond with valid JSON") == false)
        }
    }
#endif
