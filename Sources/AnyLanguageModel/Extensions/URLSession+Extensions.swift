import Foundation
import HTTPTypes
import HTTPTypesFoundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

#if canImport(FoundationNetworking)
    /// Serializes Linux URLSession operations to mitigate a FoundationNetworking race.
    ///
    /// AnyLanguageModel performs many concurrent HTTP requests across model implementations.
    /// On Linux, `FoundationNetworking` routes `URLSession` through a shared
    /// `_MultiHandle`, which has a known thread-safety bug that can crash under
    /// concurrent access (`URLSession._MultiHandle.endOperation(for:)`).
    ///
    /// This gate intentionally allows only one in-flight request setup path at a time on Linux.
    /// For non-streaming requests, callers typically hold this lock for the entire
    /// request/response cycle, effectively serializing those operations and reducing
    /// request-level parallelism (which can lower throughput for heavily concurrent
    /// workloads).
    ///
    /// For streaming requests, callers usually acquire the gate only during initial
    /// request setup and then release it once the stream has been established; stream
    /// consumption itself is not serialized by this gate.
    /// Keep this scoped to Linux-only code paths until the upstream issue is resolved.
    ///
    /// See: https://github.com/swiftlang/swift-corelibs-foundation/issues/4791
    actor LinuxURLSessionRequestGate {
        private struct Waiter {
            let id: UUID
            let continuation: CheckedContinuation<Void, Error>
        }

        static let shared = LinuxURLSessionRequestGate()

        private var isLocked = false
        private var waiters: [Waiter] = []

        func acquire() async throws {
            if Task.isCancelled {
                throw CancellationError()
            }

            if !isLocked {
                isLocked = true
                return
            }

            let waiterID = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    waiters.append(Waiter(id: waiterID, continuation: continuation))
                }
            } onCancel: {
                Task {
                    await self.cancelWaiter(id: waiterID)
                }
            }
        }

        func release() {
            if waiters.isEmpty {
                isLocked = false
                return
            }

            let waiter = waiters.removeFirst()
            waiter.continuation.resume()
        }

        private func cancelWaiter(id: UUID) {
            guard let index = waiters.firstIndex(where: { $0.id == id }) else {
                return
            }

            let waiter = waiters.remove(at: index)
            waiter.continuation.resume(throwing: CancellationError())
        }

    }

    func withLinuxRequestLock(
        _ operation: () async throws -> Void
    ) async throws {
        let gate = LinuxURLSessionRequestGate.shared
        try await gate.acquire()
        do {
            try await operation()
            await gate.release()
        } catch {
            await gate.release()
            throw error
        }
    }
#endif

/// A URL session sends a model's requests with its own configuration, reading each response's
/// body as it arrives.
extension URLSession: HTTPTransport {
    public func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        guard var urlRequest = URLRequest(httpRequest: request) else {
            throw HTTPTransportError(kind: .invalidRequest)
        }
        urlRequest.httpBody = body
        // With URLSession, it's how long to wait for each part of the response.
        let (seconds, attoseconds) = timeout.components
        urlRequest.timeoutInterval = TimeInterval(seconds) + TimeInterval(attoseconds) / 1e18

        let delegate = ResponseChunks()
        // The request is over once the handler returns: what's left of the body isn't read.
        defer { delegate.end() }
        #if canImport(FoundationNetworking)
            // Each request has a session of its own, as FoundationNetworking's shared state isn't
            // safe for requests started at once; see `LinuxURLSessionRequestGate`.
            var urlResponse: URLResponse?
            try await withLinuxRequestLock {
                let session = URLSession(configuration: self.configuration, delegate: delegate, delegateQueue: nil)
                urlResponse = try await delegate.start(session.dataTask(with: urlRequest), ownedSession: session)
            }
        #else
            let task = dataTask(with: urlRequest)
            task.delegate = delegate
            let urlResponse: URLResponse? = try await delegate.start(task, ownedSession: nil)
        #endif
        guard let response = (urlResponse as? HTTPURLResponse)?.httpResponse else {
            throw HTTPTransportError(kind: .invalidResponse)
        }
        return try await responseHandler(response, HTTPResponseBody(delegate.chunks))
    }
}

/// A data task's response and the chunks of its body, as they arrive.
private final class ResponseChunks: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private struct State {
        var responseContinuation: CheckedContinuation<URLResponse, Error>?
        var chunksContinuation: AsyncThrowingStream<Data, Error>.Continuation?
        var task: URLSessionDataTask?
        /// The session made for this request alone, which ends with it.
        var ownedSession: URLSession?
        var isCancelled = false
    }

    // Read and written from the caller's task and from the delegate's queue.
    private let state = Locked(State())

    /// The body's chunks: ending them early cancels the request.
    let chunks: AsyncThrowingStream<Data, Error>

    override init() {
        var continuation: AsyncThrowingStream<Data, Error>.Continuation!
        chunks = AsyncThrowingStream { continuation = $0 }
        super.init()
        state.withLock { $0.chunksContinuation = continuation }
        continuation.onTermination = { [weak self] _ in self?.end() }
    }

    /// Starts `task` and waits for its response. Cancelling the calling task cancels the request.
    func start(_ task: URLSessionDataTask, ownedSession: URLSession?) async throws -> URLResponse {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let isCancelled = state.withLock { state in
                    state.task = task
                    state.ownedSession = ownedSession
                    guard !state.isCancelled else { return true }
                    state.responseContinuation = continuation
                    return false
                }
                guard !isCancelled else {
                    end()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                task.resume()
            }
        } onCancel: {
            let continuation = state.withLock { state in
                state.isCancelled = true
                return Self.take(&state.responseContinuation)
            }
            end()
            continuation?.resume(throwing: CancellationError())
        }
    }

    /// Ends the request: cancels it if it's still running, and finishes the body.
    func end() {
        let (task, session, chunks) = state.withLock { state in
            defer {
                state.task = nil
                state.ownedSession = nil
            }
            return (state.task, state.ownedSession, Self.take(&state.chunksContinuation))
        }
        task?.cancel()
        session?.invalidateAndCancel()
        chunks?.finish()
    }

    /// Returns `value` at most once.
    private static func take<Value>(_ value: inout Value?) -> Value? {
        defer { value = nil }
        return value
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        state.withLock { Self.take(&$0.responseContinuation) }?.resume(returning: response)
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        _ = state.withLock { $0.chunksContinuation }?.yield(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let (response, chunks, owned) = state.withLock { state in
            defer { state.task = nil }
            return (
                Self.take(&state.responseContinuation), Self.take(&state.chunksContinuation),
                Self.take(&state.ownedSession)
            )
        }
        if let response {
            if let error {
                response.resume(throwing: error)
            } else if let urlResponse = task.response {
                response.resume(returning: urlResponse)
            } else {
                response.resume(throwing: HTTPTransportError(kind: .invalidResponse))
            }
        }
        if let error {
            chunks?.finish(throwing: error)
        } else {
            chunks?.finish()
        }
        owned?.finishTasksAndInvalidate()
    }
}
