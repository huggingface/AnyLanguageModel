import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Code written for 1.0 still builds, and the code each deprecation's fix-it turns it into builds
/// too: a fix-it that doesn't compile leaves its user with an error instead of a warning.
@Suite("Migrating from 1.0")
struct MigrationTests {
    private static let url = URL(string: "http://localhost:1")!

    /// As 1.0 wrote it, with a session.
    @available(*, deprecated)
    @Test func codeWrittenFor1_0Builds() {
        let session: HTTPSession = makeDefaultSession()
        _ = OllamaLanguageModel(baseURL: Self.url, model: "m", session: session)
        _ = OpenAILanguageModel(baseURL: Self.url, apiKey: "k", model: "m", session: session)
        _ = AnthropicLanguageModel(baseURL: Self.url, apiKey: "k", model: "m", session: session)
        _ = GeminiLanguageModel(baseURL: Self.url, apiKey: "k", model: "m", session: session)
        _ = OpenResponsesLanguageModel(baseURL: Self.url, apiKey: "k", model: "m", session: session)
    }

    /// As each `renamed:` fix-it rewrites it: the session given as the transport.
    @available(*, deprecated)
    @Test func codeTheFixItsWriteBuilds() {
        let session: HTTPSession = makeDefaultSession()
        _ = OllamaLanguageModel(baseURL: Self.url, model: "m", transport: session)
        _ = OpenAILanguageModel(baseURL: Self.url, apiKey: "k", model: "m", transport: session)
        _ = AnthropicLanguageModel(baseURL: Self.url, apiKey: "k", model: "m", transport: session)
        _ = GeminiLanguageModel(baseURL: Self.url, apiKey: "k", model: "m", transport: session)
        _ = OpenResponsesLanguageModel(baseURL: Self.url, apiKey: "k", model: "m", transport: session)
    }

    /// As the deprecation of `makeDefaultSession()` says to write it.
    @Test func theDefaultTransportBuilds() {
        let transport: any HTTPTransport = makeDefaultTransport()
        _ = OllamaLanguageModel(baseURL: Self.url, model: "m", transport: transport)
    }
}
