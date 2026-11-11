import Foundation
import HTTPTypes
import Testing

@testable import AnyLanguageModel

/// A provider's models are asked for as its responses are, with its address, key and transport:
/// by the IDs a model is made with, every page of them; a refused key throws the provider's answer,
/// as a response's does.
@Suite("Available models")
struct AvailableModelsTests {
    @Test("OpenAI and servers compatible with it list their models, asked with a key or none")
    func openAI() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [#"{"object":"list","data":[{"id":"gpt-5","object":"model"},{"id":"gpt-5-mini"}]}"#]),
            .init(chunks: [#"{"data":[{"id":"qwen3"}]}"#]),
        ])

        let models = try await OpenAILanguageModel.availableModels(apiKey: "secret", transport: transport)
        let local = try await OpenAILanguageModel.availableModels(
            baseURL: URL(string: "http://localhost:1234/v1")!,
            apiKey: nil,
            transport: transport
        )

        #expect(models == ["gpt-5", "gpt-5-mini"])
        #expect(local == ["qwen3"])
        #expect(transport.requests.map(\.method) == [.get, .get])
        #expect(transport.requests.map(\.path) == ["/v1/models", "/v1/models"])
        #expect(transport.requests.map { $0.headerFields[.authorization] } == ["Bearer secret", nil])
    }

    @Test("Anthropic's models are every page of them, asked with its key and version")
    func anthropic() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [#"{"data":[{"id":"claude-a"},{"id":"claude-b"}],"has_more":true,"last_id":"claude-b"}"#]),
            .init(chunks: [#"{"data":[{"id":"claude-c"}],"has_more":false,"last_id":"claude-c"}"#]),
        ])

        let models = try await AnthropicLanguageModel.availableModels(apiKey: "secret", transport: transport)

        #expect(models == ["claude-a", "claude-b", "claude-c"])
        #expect(
            transport.requests.map(\.path) == ["/v1/models?limit=1000", "/v1/models?limit=1000&after_id=claude-b"]
        )
        let headers = try #require(transport.requests.first?.headerFields)
        #expect(headers[HTTPField.Name("x-api-key")!] == "secret")
        #expect(headers[HTTPField.Name("anthropic-version")!] == AnthropicLanguageModel.defaultAPIVersion)
    }

    @Test("a page whose cursor doesn't move is the last, so listing ends")
    func anthropicStopsWhenTheCursorStays() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [#"{"data":[{"id":"claude-a"}],"has_more":true,"last_id":"claude-a"}"#]),
            .init(chunks: [#"{"data":[],"has_more":true,"last_id":"claude-a"}"#]),
        ])

        let models = try await AnthropicLanguageModel.availableModels(apiKey: "secret", transport: transport)

        #expect(models == ["claude-a"])
        #expect(transport.requests.count == 2)
    }

    @Test("Gemini's models are those that generate content, by their IDs, every page of them")
    func gemini() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [
                #"{"models":[{"name":"models/gemini-pro","supportedGenerationMethods":["generateContent","countTokens"]},{"name":"models/embedding","supportedGenerationMethods":["embedContent"]}],"nextPageToken":"next"}"#
            ]),
            .init(chunks: [
                #"{"models":[{"name":"models/gemini-flash","supportedGenerationMethods":["generateContent"]}]}"#
            ]),
            .init(chunks: ["{}"]),
        ])

        let models = try await GeminiLanguageModel.availableModels(apiKey: "secret", transport: transport)
        let none = try await GeminiLanguageModel.availableModels(apiKey: "secret", transport: transport)

        #expect(models == ["gemini-pro", "gemini-flash"])
        #expect(none.isEmpty)
        #expect(
            transport.requests.prefix(2).map(\.path) == [
                "/v1beta/models?pageSize=1000", "/v1beta/models?pageSize=1000&pageToken=next",
            ]
        )
        #expect(transport.requests.first?.headerFields[HTTPField.Name("x-goog-api-key")!] == "secret")
    }

    @Test("an Ollama server lists the models it has, and those it's running, with its key or none")
    func ollama() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [#"{"models":[{"name":"qwen3:8b","model":"qwen3:8b"},{"name":"gpt-oss:120b"}]}"#]),
            .init(chunks: [#"{"models":[{"name":"qwen3:8b"}]}"#]),
        ])
        let cloud = URL(string: "https://ollama.com")!

        let available = try await OllamaLanguageModel.availableModels(transport: transport)
        let running = try await OllamaLanguageModel.runningModels(
            baseURL: cloud,
            apiKey: "secret",
            transport: transport
        )

        #expect(available == ["qwen3:8b", "gpt-oss:120b"])
        #expect(running == ["qwen3:8b"])
        #expect(transport.requests.map(\.path) == ["/api/tags", "/api/ps"])
        #expect(transport.requests.map { $0.headerFields[.authorization] } == [nil, "Bearer secret"])
    }

    @Test(
        "a refused key throws the provider's answer, as a response's does",
        arguments: [Provider.openAI, .anthropic, .gemini, .ollama]
    )
    func aRefusedKeyThrowsTheAnswer(provider: Provider) async throws {
        let refusal = #"{"error":{"message":"invalid key"}}"#
        let transport = FakeTransport(replies: [.init(status: provider.refusal, chunks: [refusal])])

        let error = try await #require(throws: HTTPTransportError.self) {
            _ = try await provider.availableModels(transport)
        }

        #expect(error.kind == .httpError)
        #expect(error.response?.status == provider.refusal)
        #expect(error.body == Data(refusal.utf8))
    }

    @Test("a page that fails partway through fails the whole listing")
    func aFailingPageFailsTheListing() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [#"{"data":[{"id":"claude-a"}],"has_more":true,"last_id":"claude-a"}"#]),
            .init(status: .internalServerError, chunks: ["{}"]),
        ])

        let error = try await #require(throws: HTTPTransportError.self) {
            _ = try await AnthropicLanguageModel.availableModels(apiKey: "secret", transport: transport)
        }

        #expect(error.response?.status == .internalServerError)
    }

    @Test("an answer that isn't a list of models is a decoding error")
    func anUnreadableAnswerIsADecodingError() async throws {
        let transport = FakeTransport(replies: [.init(chunks: [#"{"unexpected":true}"#])])

        let error = try await #require(throws: HTTPTransportError.self) {
            _ = try await OpenAILanguageModel.availableModels(apiKey: "secret", transport: transport)
        }

        #expect(error.kind == .decodingError)
    }

    @Test("a listing cancelled while a page is asked for asks for no other")
    func cancellingStopsTheNextPage() async throws {
        let transport = WaitingForCancellation(
            first: #"{"data":[{"id":"claude-a"}],"has_more":true,"last_id":"claude-a"}"#
        )
        let listing = Task { try await AnthropicLanguageModel.availableModels(apiKey: "secret", transport: transport) }
        while !transport.started { try await Task.sleep(for: .milliseconds(1)) }
        listing.cancel()

        await #expect(throws: CancellationError.self) { try await listing.value }
        #expect(transport.sent == 1)
    }

    enum Provider: CaseIterable, Sendable {
        case openAI, anthropic, gemini, ollama

        /// How it refuses a key: Gemini answers one that isn't valid with `400 Bad Request`.
        var refusal: HTTPResponse.Status {
            self == .gemini ? .badRequest : .unauthorized
        }

        func availableModels(_ transport: FakeTransport) async throws -> [String] {
            switch self {
            case .openAI: try await OpenAILanguageModel.availableModels(apiKey: "wrong", transport: transport)
            case .anthropic: try await AnthropicLanguageModel.availableModels(apiKey: "wrong", transport: transport)
            case .gemini: try await GeminiLanguageModel.availableModels(apiKey: "wrong", transport: transport)
            case .ollama:
                try await OllamaLanguageModel.runningModels(
                    baseURL: URL(string: "https://ollama.com")!,
                    apiKey: "wrong",
                    transport: transport
                )
            }
        }
    }
}

/// A transport whose first answer waits until its task is cancelled, then comes as if nothing
/// happened: a transport that doesn't stop for cancellation.
private final class WaitingForCancellation: HTTPTransport {
    private let first: String
    private let state = Locked((started: false, sent: 0))

    init(first: String) {
        self.first = first
    }

    var started: Bool { state.withLock { $0.started } }
    var sent: Int { state.withLock { $0.sent } }

    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        state.withLock {
            $0.started = true
            $0.sent += 1
        }
        while !Task.isCancelled { await Task.yield() }
        let chunks = AsyncStream<Data> { continuation in
            continuation.yield(Data(first.utf8))
            continuation.finish()
        }
        return try await responseHandler(HTTPResponse(status: .ok), HTTPResponseBody(chunks))
    }
}
