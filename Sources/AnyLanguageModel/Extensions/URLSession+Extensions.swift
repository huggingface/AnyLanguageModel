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
    public func perform<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        guard var urlRequest = URLRequest(httpRequest: request) else { throw HTTPTransportError.invalidRequest }
        urlRequest.httpBody = body
        // With URLSession, it's how long to wait for each part of the response.
        let (seconds, attoseconds) = timeout.components
        urlRequest.timeoutInterval = TimeInterval(seconds) + TimeInterval(attoseconds) / 1e18

        #if canImport(FoundationNetworking)
            var started: (bytes: AsyncThrowingStream<UInt8, Error>, response: URLResponse)?
            try await withLinuxRequestLock {
                started = try await self.linuxBytes(for: urlRequest)
            }
            guard let started, let response = (started.response as? HTTPURLResponse)?.httpResponse else { throw HTTPTransportError.invalidResponse }
            return try await responseHandler(response, HTTPResponseBody(started.bytes))
        #else
            let (bytes, urlResponse) = try await self.bytes(for: urlRequest)
            guard let response = (urlResponse as? HTTPURLResponse)?.httpResponse else { throw HTTPTransportError.invalidResponse }
            return try await responseHandler(response, HTTPResponseBody(bytes))
        #endif
    }
}

#if canImport(FoundationNetworking)
    private extension URLSession {
        func linuxBytes(for request: URLRequest) async throws -> (AsyncThrowingStream<UInt8, Error>, URLResponse) {
            let delegate = LinuxBytesDelegate()
            let delegateQueue = OperationQueue()
            delegateQueue.maxConcurrentOperationCount = 1

            let session = URLSession(
                configuration: self.configuration,
                delegate: delegate,
                delegateQueue: delegateQueue
            )

            let byteStream = AsyncThrowingStream<UInt8, Error> { continuation in
                delegate.attach(
                    continuation,
                    session: session
                )
            }

            let response = try await delegate.start(
                request: request,
                session: session
            )

            return (byteStream, response)
        }
    }

    private final class LinuxBytesDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private struct State {
            var responseContinuation: CheckedContinuation<URLResponse, Error>?
            var isCancelled = false
        }

        // Read and written from the caller's task and from the delegate queue.
        private let state = Locked(State())
        private var byteContinuation: AsyncThrowingStream<UInt8, Error>.Continuation?
        private weak var task: URLSessionDataTask?
        private weak var session: URLSession?

        func attach(
            _ continuation: AsyncThrowingStream<UInt8, Error>.Continuation,
            session: URLSession
        ) {
            byteContinuation = continuation
            self.session = session
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.task?.cancel()
                self.session?.invalidateAndCancel()
            }
        }

        /// Starts the request and waits for its response.
        /// Canceling the calling task cancels the request.
        func start(
            request: URLRequest,
            session: URLSession
        ) async throws -> URLResponse {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    let task = session.dataTask(with: request)
                    let isCancelled = state.withLock { state in
                        guard !state.isCancelled else { return true }
                        state.responseContinuation = continuation
                        return false
                    }
                    guard !isCancelled else {
                        session.invalidateAndCancel()
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    self.task = task
                    task.resume()
                }
            } onCancel: {
                cancel(session: session)
            }
        }

        private func cancel(session: URLSession) {
            let continuation = state.withLock { state in
                state.isCancelled = true
                return takeResponseContinuation(&state)
            }
            session.invalidateAndCancel()
            continuation?.resume(throwing: CancellationError())
        }

        /// Returns the response continuation at most once,
        /// so it's resumed exactly once by whichever of the response, completion, or cancellation comes first.
        private func takeResponseContinuation(_ state: inout State) -> CheckedContinuation<URLResponse, Error>? {
            defer { state.responseContinuation = nil }
            return state.responseContinuation
        }

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
        ) {
            state.withLock { takeResponseContinuation(&$0) }?.resume(returning: response)
            completionHandler(.allow)
        }

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive data: Data
        ) {
            guard let continuation = byteContinuation else { return }
            for byte in data {
                continuation.yield(byte)
            }
        }

        func urlSession(
            _ session: URLSession,
            task: URLSessionTask,
            didCompleteWithError error: (any Error)?
        ) {
            if let continuation = state.withLock({ takeResponseContinuation(&$0) }) {
                if let error {
                    continuation.resume(throwing: error)
                } else if let response = task.response {
                    continuation.resume(returning: response)
                } else {
                    continuation.resume(throwing: HTTPTransportError.invalidResponse)
                }
            }

            if let error {
                byteContinuation?.finish(throwing: error)
            } else {
                byteContinuation?.finish()
            }
            byteContinuation = nil

            session.invalidateAndCancel()
        }
    }
#endif
