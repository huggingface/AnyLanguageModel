import Foundation
import HTTPTypes
import Testing

@testable import AnyLanguageModel

/// Whether a model can be used now, asked where it runs: a provider says whether it takes the key
/// and has the model; a model that asks no one says what its availability says.
@Suite("Checking availability")
struct CheckAvailabilityTests {
    @Test("a model that asks no one says what its availability says")
    func theDefaultIsTheModelsAvailability() async throws {
        var model = MockLanguageModel()
        #expect(try await model.checkAvailability().kind == nil)

        model.availabilityProvider = { .unavailable(.custom("Not today")) }
        #expect(try await model.checkAvailability().kind == .other)
    }

    @Test("an Ollama server takes the key when it says what it runs, then shows the model")
    func ollamaIsAvailable() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [#"{"models":[]}"#]),
            .init(chunks: [#"{"modelfile":"FROM qwen3"}"#]),
        ])
        let model = OllamaLanguageModel(
            baseURL: URL(string: "https://ollama.com")!,
            apiKey: "secret",
            model: "qwen3",
            transport: transport
        )

        #expect(try await model.checkAvailability().kind == nil)
        #expect(transport.requests.map(\.path) == ["/api/ps", "/api/show"])
        #expect(transport.requests.map { $0.headerFields[.authorization] } == ["Bearer secret", "Bearer secret"])
        #expect(transport.bodies.last == Data(#"{"model":"qwen3"}"#.utf8))
    }

    @Test(
        "an Ollama server that refuses the key, or has no such model, says so",
        arguments: [
            (
                key: "secret", status: HTTPResponse.Status.unauthorized, show: false,
                kind: ModelUnavailability.Kind.keyRefused
            ),
            (key: nil, status: .unauthorized, show: false, kind: .keyNeeded),
            (key: "secret", status: .notFound, show: true, kind: .modelNotFound),
            (key: "secret", status: .notFound, show: false, kind: .providerError),
            (key: "secret", status: .internalServerError, show: true, kind: .providerError),
        ] as [(key: String?, status: HTTPResponse.Status, show: Bool, kind: ModelUnavailability.Kind)]
    )
    func ollamaIsUnavailable(
        _ outcome: (key: String?, status: HTTPResponse.Status, show: Bool, kind: ModelUnavailability.Kind)
    ) async throws {
        // A refusal of `api/ps` comes first; of `api/show`, after `api/ps` took the key.
        let transport = FakeTransport(
            replies: (outcome.show ? [.init(chunks: [#"{"models":[]}"#])] : [])
                + [.init(status: outcome.status, chunks: [#"{"error":"no"}"#])]
        )
        let cloud = URL(string: "https://ollama.com")!
        let model =
            if let key = outcome.key {
                OllamaLanguageModel(baseURL: cloud, apiKey: key, model: "qwen3", transport: transport)
            } else {
                OllamaLanguageModel(baseURL: cloud, model: "qwen3", transport: transport)
            }

        let unavailability = try #require(try await model.checkAvailability().unavailability)

        #expect(unavailability.kind == outcome.kind)
        #expect((unavailability.underlyingError as? HTTPTransportError)?.response?.status == outcome.status)
    }

    @Test("OpenAI and servers compatible with it are available when they list the model")
    func openAI() async throws {
        let listing = #"{"data":[{"id":"gpt-5"}]}"#
        let transport = FakeTransport(replies: [.init(chunks: [listing]), .init(chunks: [listing])])
        let listed = OpenAILanguageModel(apiKey: "secret", model: "gpt-5", transport: transport)
        let unlisted = OpenAILanguageModel(
            baseURL: URL(string: "http://localhost:1234/v1")!,
            apiKey: "",
            model: "gpt-6",
            transport: transport
        )

        #expect(try await listed.checkAvailability().kind == nil)
        #expect(try await unlisted.checkAvailability().kind == .modelNotFound)
        // A server that takes no key is asked without one.
        #expect(transport.requests.map { $0.headerFields[.authorization] } == ["Bearer secret", nil])
    }

    @Test("an Open Responses server is available when it lists the model")
    func openResponses() async throws {
        let transport = FakeTransport(replies: [.init(chunks: [#"{"data":[{"id":"m"}]}"#])])
        let model = OpenResponsesLanguageModel(
            baseURL: URL(string: "http://localhost:8000/v1")!,
            apiKey: "secret",
            model: "m",
            transport: transport
        )

        #expect(try await model.checkAvailability().kind == nil)
        #expect(transport.requests.map(\.path) == ["/v1/models"])
    }

    @Test(
        "Anthropic is asked for the model itself: a refused key, or no such model, says so",
        arguments: [
            (HTTPResponse.Status.ok, ModelUnavailability.Kind?.none),
            (.unauthorized, .keyRefused),
            (.notFound, .modelNotFound),
            (.internalServerError, .providerError),
        ] as [(HTTPResponse.Status, ModelUnavailability.Kind?)]
    )
    func anthropic(status: HTTPResponse.Status, kind: ModelUnavailability.Kind?) async throws {
        let transport = FakeTransport(replies: [.init(status: status, chunks: [#"{"id":"claude-sonnet"}"#])])
        let model = AnthropicLanguageModel(
            apiKey: "secret",
            betas: ["beta"],
            model: "claude-sonnet",
            transport: transport
        )

        #expect(try await model.checkAvailability().kind == kind)
        #expect(transport.requests.map(\.path) == ["/v1/models/claude-sonnet"])
        #expect(transport.requests.first?.headerFields[HTTPField.Name("anthropic-beta")!] == "beta")
    }

    @Test(
        "Gemini is asked for the model itself; a key it says isn't valid is refused",
        arguments: [
            (HTTPResponse.Status.ok, #"{"name":"models/gemini-pro"}"#, ModelUnavailability.Kind?.none),
            (.badRequest, #"{"error":{"code":400,"details":[{"reason":"API_KEY_INVALID"}]}}"#, .keyRefused),
            (.badRequest, #"{"error":{"code":400,"details":[{"reason":"OTHER"}]}}"#, .providerError),
            (.forbidden, #"{"error":{"code":403}}"#, .keyRefused),
            (.notFound, #"{"error":{"code":404}}"#, .modelNotFound),
        ] as [(HTTPResponse.Status, String, ModelUnavailability.Kind?)]
    )
    func gemini(status: HTTPResponse.Status, answer: String, kind: ModelUnavailability.Kind?) async throws {
        let transport = FakeTransport(replies: [.init(status: status, chunks: [answer])])
        let model = GeminiLanguageModel(apiKey: "secret", model: "gemini-pro", transport: transport)

        #expect(try await model.checkAvailability().kind == kind)
        #expect(transport.requests.map(\.path) == ["/v1beta/models/gemini-pro"])
    }

    @Test("a provider that can't be reached says so, with why")
    func anUnreachableProviderSaysSo() async throws {
        let model = AnthropicLanguageModel(apiKey: "secret", model: "m", transport: Unreachable())

        let unavailability = try #require(try await model.checkAvailability().unavailability)

        #expect(unavailability.kind == .unreachable)
        #expect((unavailability.underlyingError as? URLError)?.code == .cannotConnectToHost)
    }

    @Test("a check cancelled while it asks throws, neither available nor not")
    func aCancelledCheckThrows() async throws {
        let transport = WaitingForCancellation(answer: #"{"id":"m"}"#)
        let check = Task {
            try await AnthropicLanguageModel(apiKey: "secret", model: "m", transport: transport).checkAvailability()
        }
        while !transport.started { try await Task.sleep(for: .milliseconds(1)) }
        check.cancel()

        await #expect(throws: CancellationError.self) { try await check.value }
    }

    #if canImport(FoundationModels) && (!os(tvOS) || compiler(<6.4))
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @available(watchOS, unavailable)
        @Test("the system model says what the system says")
        func systemReasons() {
            #expect(SystemLanguageModel.kind(of: .deviceNotEligible) == .deviceNotEligible)
            #expect(SystemLanguageModel.kind(of: .appleIntelligenceNotEnabled) == .appleIntelligenceNotEnabled)
            #expect(SystemLanguageModel.kind(of: .modelNotReady) == .notReady)
        }
    #endif

    #if Llama
        @Test("a llama.cpp model is available when its files are there")
        func llamaFiles() async throws {
            let file = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).gguf")
            let missing = LlamaLanguageModel(modelPath: file.path)
            #expect(try await missing.checkAvailability().kind == .modelNotFound)

            try Data().write(to: file)
            defer { try? FileManager.default.removeItem(at: file) }
            #expect(try await LlamaLanguageModel(modelPath: file.path).checkAvailability().kind == nil)
            let projector = LlamaLanguageModel(modelPath: file.path, mmprojPath: file.path + ".missing")
            #expect(try await projector.checkAvailability().kind == .modelNotFound)
        }
    #endif
}

/// A transport whose requests fail on the way, as one to a server that isn't there does.
private struct Unreachable: HTTPTransport {
    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        throw URLError(.cannotConnectToHost)
    }
}

extension ModelAvailability {
    /// Why it's unavailable, when it is.
    fileprivate var unavailability: ModelUnavailability? {
        if case .unavailable(let unavailability) = self { unavailability } else { nil }
    }

    /// What's in the way, when something is.
    fileprivate var kind: ModelUnavailability.Kind? { unavailability?.kind }
}
