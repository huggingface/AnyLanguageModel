import Foundation
import HTTPTypes
import Testing

@testable import AnyLanguageModel

/// An Ollama model made with an API key sends it as a bearer token with each request, as Ollama's
/// cloud asks; one made without a key sends no `Authorization`, as a local server asks for none.
@Suite("Ollama authorization")
struct OllamaAuthorizationTests {
    @Test("without an API key, no Authorization is sent", arguments: [false, true])
    func withoutAKeyNoAuthorizationIsSent(streaming: Bool) async throws {
        let transport = ollama(replies: 1)
        let model = OllamaLanguageModel(model: "m", transport: transport)

        try await respond(LanguageModelSession(model: model), streaming: streaming)

        #expect(transport.requests.count == 1)
        #expect(transport.requests.first?.headerFields[.authorization] == nil)
    }

    @Test("an API key is sent as a bearer token", arguments: [false, true])
    func aKeyIsSentAsABearerToken(streaming: Bool) async throws {
        let transport = ollama(replies: 1)
        let model = OllamaLanguageModel(apiKey: "secret", model: "m", transport: transport)

        try await respond(LanguageModelSession(model: model), streaming: streaming)

        #expect(transport.requests.map { $0.headerFields[.authorization] } == ["Bearer secret"])
    }

    @Test("the API key is asked for again for each request", arguments: [false, true])
    func theKeyIsAskedForEachRequest(streaming: Bool) async throws {
        let transport = ollama(replies: 2)
        let asked = Locked(0)
        @Sendable func nextKey() -> String {
            asked.withLock { asked in
                asked += 1
                return "key-\(asked)"
            }
        }
        let model = OllamaLanguageModel(apiKey: nextKey(), model: "m", transport: transport)
        let session = LanguageModelSession(model: model)

        try await respond(session, streaming: streaming)
        try await respond(session, streaming: streaming)

        #expect(transport.requests.map { $0.headerFields[.authorization] } == ["Bearer key-1", "Bearer key-2"])
    }

    @Test("each tool round's request sends the API key", arguments: [false, true])
    func eachToolRoundSendsTheKey(streaming: Bool) async throws {
        let call =
            #"{"model":"m","created_at":"2026-11-11T10:00:00.000Z","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_weather","type":"function","function":{"name":"getWeather","arguments":{"city":"Munich"}}}]},"done":\#(!streaming)}"#
        let done =
            #"{"model":"m","created_at":"2026-11-11T10:00:00.000Z","message":{"role":"assistant","content":""},"done":true}"#
        let transport = FakeTransport(replies: [
            .init(chunks: [(streaming ? [call, done] : [call]).joined(separator: "\n") + "\n"]),
            .init(chunks: [reply + "\n"]),
        ])
        let model = OllamaLanguageModel(apiKey: "secret", model: "m", transport: transport)

        try await respond(LanguageModelSession(model: model, tools: [WeatherTool()]), streaming: streaming)

        #expect(transport.requests.map { $0.headerFields[.authorization] } == ["Bearer secret", "Bearer secret"])
    }

    @Test(
        "a refused API key, or an empty one, fails with the server's answer, and the next request goes on",
        arguments: ["wrong", ""],
        [false, true]
    )
    func aRefusedKeyFailsAndTheNextRequestGoesOn(key: String, streaming: Bool) async throws {
        let refusal = #"{"error":"unauthorized"}"#
        let transport = FakeTransport(replies: [
            .init(status: .unauthorized, chunks: [refusal]),
            .init(chunks: [reply + "\n"]),
        ])
        let model = OllamaLanguageModel(apiKey: key, model: "m", transport: transport)
        let session = LanguageModelSession(model: model)

        let error = try await #require(throws: HTTPTransportError.self) {
            try await respond(session, streaming: streaming)
        }
        #expect(error.response?.status == .unauthorized)
        try await respond(session, streaming: streaming)

        #expect(transport.requests.count == 2)
        #expect(transport.requests.allSatisfy { $0.headerFields[.authorization] != nil })
    }
}

private let reply =
    #"{"model":"m","created_at":"2026-11-11T10:00:00.000Z","message":{"role":"assistant","content":"Hi"},"done":true}"#

/// A transport that answers `replies` requests, each with one finished reply.
private func ollama(replies: Int) -> FakeTransport {
    FakeTransport(replies: Array(repeating: .init(chunks: [reply + "\n"]), count: replies))
}

/// `session` asked once, streamed or not.
private func respond(_ session: LanguageModelSession, streaming: Bool) async throws {
    if streaming {
        _ = try await session.streamResponse(to: "Hello").collect()
    } else {
        _ = try await session.respond(to: "Hello")
    }
}
