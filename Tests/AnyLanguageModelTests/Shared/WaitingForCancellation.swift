import Foundation
import HTTPTypes

@testable import AnyLanguageModel

/// A transport whose first answer waits until its task is cancelled, then comes as if nothing
/// happened: a transport that doesn't stop for cancellation.
final class WaitingForCancellation: HTTPTransport {
    private let answer: String
    private let state = Locked((started: false, sent: 0))

    init(answer: String) {
        self.answer = answer
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
            continuation.yield(Data(answer.utf8))
            continuation.finish()
        }
        return try await responseHandler(HTTPResponse(status: .ok), HTTPResponseBody(chunks))
    }
}
