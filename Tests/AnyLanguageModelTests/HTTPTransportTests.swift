import Foundation
import HTTPTypes
import HTTPTypesFoundation
import Testing

#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
#endif

@testable import AnyLanguageModel

@Suite("HTTPTransport")
struct HTTPTransportTests {
    static let ollama = URL(string: "http://localhost:11434")!

    @Test func modelSendsItsRequestThroughTheTransport() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [
                #"{"model":"m","created_at":"2026-10-18T10:00:00.000Z","mess"#,
                #"age":{"role":"assistant","content":"Hello"},"done":true}"#,
            ])
        ])
        let model = OllamaLanguageModel(baseURL: Self.ollama, model: "m", transport: transport)
        let response = try await LanguageModelSession(model: model).respond(to: "Hi")

        #expect(response.content == "Hello")
        let request = try #require(transport.requests.first)
        #expect(request.method == .post)
        #expect(request.path == "/api/chat")
        #expect(request.headerFields[.contentType] == "application/json")
        #expect(String(decoding: try #require(transport.bodies.first), as: UTF8.self).contains(#""model":"m""#))
        #expect(transport.handled == 1)
    }

    @Test func streamedLinesSplitAcrossChunksAreRead() async throws {
        let transport = FakeTransport(replies: [
            .init(chunks: [
                #"{"model":"m","created_at":"2026-10-18T10:00:00.000Z","message":{"role":"assistant","content":"Hel"},"done":false}"#
                    + "\n" + #"{"model":"m","created_at":"2026-10-18T10:00:00.000Z","message":{"role":"ass"#,
                #"istant","content":"lo"},"done":false}"# + "\n",
                #"{"model":"m","created_at":"2026-10-18T10:00:00.000Z","message":{"role":"assistant","content":""},"done":true}"#,
            ])
        ])
        let model = OllamaLanguageModel(baseURL: Self.ollama, model: "m", transport: transport)
        let response = try await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()

        #expect(response.content == "Hello")
    }

    @Test func eventsSplitAcrossChunksAreRead() async throws {
        func event(_ content: String) -> String {
            #"data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"\#(content)"},"finish_reason":null}]}"#
                + "\n\n"
        }
        let events = event("Hel") + event("lo") + "data: [DONE]\n\n"
        let middle = events.index(events.startIndex, offsetBy: events.count / 2)
        let transport = FakeTransport(replies: [.init(chunks: [String(events[..<middle]), String(events[middle...])])])
        let model = OpenAILanguageModel(apiKey: "k", model: "m", transport: transport)
        let response = try await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()

        #expect(response.content == "Hello")
    }

    @Test func errorResponseKeepsItsHeadersAndBody() async throws {
        let transport = FakeTransport(replies: [
            .init(status: .tooManyRequests, headerFields: [.retryAfter: "20"], chunks: [#"{"error":"slow down"}"#])
        ])
        let model = OllamaLanguageModel(baseURL: Self.ollama, model: "m", transport: transport)
        let error = try await #require(throws: HTTPTransportError.self) {
            try await LanguageModelSession(model: model).respond(to: "Hi")
        }
        #expect(error.kind == .httpError)
        #expect(error.response?.status == .tooManyRequests)
        #expect(error.response?.headerFields[.retryAfter] == "20")
        #expect(error.body == Data(#"{"error":"slow down"}"#.utf8))
    }

    /// A transport whose body ends quietly when it's cancelled, as an `AsyncStream` does: what
    /// arrived isn't read as a whole answer.
    @Test func aBodyCutShortByCancellingIsntDecoded() async throws {
        let model = OllamaLanguageModel(
            baseURL: Self.ollama,
            model: "m",
            transport: StalledTransport(sent: #"{"model":"#)
        )
        let task = Task { try await LanguageModelSession(model: model).respond(to: "Hi") }
        try await Task.sleep(for: .milliseconds(200))
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test func aStreamedLineThatDoesntDecodeIsADecodingError() async throws {
        let transport = FakeTransport(replies: [.init(chunks: ["not json\n"])])
        let model = OllamaLanguageModel(baseURL: Self.ollama, model: "m", transport: transport)
        let error = try await #require(throws: HTTPTransportError.self) {
            try await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()
        }
        #expect(error.kind == .decodingError)
        #expect(error.underlyingError is DecodingError)
    }

    #if canImport(AsyncHTTPClient)
        /// A timeout too long to count in nanoseconds means no limit, rather than a trap.
        @Test func aHugeTimeoutDoesntTrap() async throws {
            let server = try LoopbackServer(.partial(contentLength: 2, body: "{}"))
            defer { server.stop() }
            let body = try await HTTPClient.shared.send(
                request: HTTPRequest(url: server.url),
                body: nil,
                timeout: .seconds(Int64.max)
            ) { _, body in try await body.collect() }
            #expect(body == Data("{}".utf8))
        }
    #endif
}

/// A transport that sends the start of a body and then nothing, until it's cancelled, when its
/// body ends quietly, as an `AsyncStream`'s does.
private struct StalledTransport: HTTPTransport {
    let sent: String

    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        let chunks = AsyncStream<Data> { continuation in continuation.yield(Data(sent.utf8)) }
        return try await responseHandler(HTTPResponse(status: .ok), HTTPResponseBody(chunks))
    }
}
