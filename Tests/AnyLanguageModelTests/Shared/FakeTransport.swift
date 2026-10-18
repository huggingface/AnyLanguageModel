import Foundation
import HTTPTypes

@testable import AnyLanguageModel

/// A transport that answers each request with the next of its replies, in the chunks given, and
/// keeps what it was sent.
final class FakeTransport: HTTPTransport {
    struct Reply: Sendable {
        var status: HTTPResponse.Status = .ok
        var headerFields: HTTPFields = [:]
        var chunks: [String]
    }

    private struct State {
        var replies: [Reply]
        var requests: [HTTPRequest] = []
        var bodies: [Data] = []
        var handled = 0
    }

    private let state: Locked<State>

    init(replies: [Reply]) {
        state = Locked(State(replies: replies))
    }

    /// The requests sent, in order.
    var requests: [HTTPRequest] { state.withLock { $0.requests } }

    /// The bodies sent, in order.
    var bodies: [Data] { state.withLock { $0.bodies } }

    /// How many times a response was passed to a handler.
    var handled: Int { state.withLock { $0.handled } }

    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        let reply = state.withLock { state in
            state.requests.append(request)
            state.bodies.append(body ?? Data())
            state.handled += 1
            return state.replies.removeFirst()
        }
        let chunks = AsyncStream<Data> { continuation in
            for chunk in reply.chunks { continuation.yield(Data(chunk.utf8)) }
            continuation.finish()
        }
        return try await responseHandler(
            HTTPResponse(status: reply.status, headerFields: reply.headerFields),
            HTTPResponseBody(chunks)
        )
    }
}
