import Foundation
import HTTPTypes
#if !os(WASI)
    import HTTPTypesFoundation
#endif

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// WASI has no URLSession, so models there need a transport given to them.
#if !os(WASI)
    #if canImport(FoundationNetworking)
        /// Serializes Linux URLSession operations to mitigate a FoundationNetworking race.
        ///
        /// AnyLanguageModel performs many concurrent HTTP requests across model implementations.
        /// On Linux, `FoundationNetworking` routes `URLSession` through a shared
        /// `_MultiHandle`, which has a known thread-safety bug that can crash under
        /// concurrent access (`URLSession._MultiHandle.endOperation(for:)`).
        ///
        /// This gate intentionally allows only one request to start at a time on Linux. Every
        /// request, streamed or not, holds it only until its response head arrives; reading the
        /// body isn't serialized by it.
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
    ///
    /// It waits for each part of a response, not only for its start, as long as `timeout`, or as the
    /// session's own `timeoutIntervalForRequest` says when it isn't given. On Linux it waits whole
    /// seconds, so a `timeout` is rounded up to the next one. Cancelling the calling task cancels the request, which then throws
    /// `URLError.cancelled`, as a cancelled URL session task does.
    ///
    /// On Linux, each request is sent by a session of its own, made with this session's
    /// configuration, as FoundationNetworking can't send requests at once through one session:
    /// this session's delegate isn't used there.
    extension URLSession: HTTPTransport {
        public func send<Return>(
            request: HTTPRequest,
            body: Data?,
            timeout: Duration?,
            responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
        ) async throws -> Return {
            guard var urlRequest = URLRequest(httpRequest: request) else {
                throw HTTPTransportError(kind: .invalidRequest)
            }
            urlRequest.httpBody = body
            // With URLSession, it's how long to wait for each part of the response. A request's own
            // timeout takes the place of the session's, so it's set only when the caller gave one.
            // FoundationNetworking schedules it with Dispatch, which counts it in nanoseconds from now
            // in an `Int` and traps past that, so it's at most about 68 years: no limit to speak of, as
            // with AsyncHTTPClient.
            if let (seconds, attoseconds) = timeout?.components {
                var interval = TimeInterval(seconds) + TimeInterval(attoseconds) / 1e18
                #if canImport(FoundationNetworking)
                    // FoundationNetworking keeps only whole seconds of `timeoutInterval`, so half a
                    // second would be none: it's rounded up.
                    interval = max(1, interval.rounded(.up))
                #endif
                urlRequest.timeoutInterval = min(interval, TimeInterval(Int32.max))
            }

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
            return try await responseHandler(response, HTTPResponseBody(delegate))
        }
    }

    /// A data task's response and the chunks of its body, as they arrive.
    ///
    /// Cancelling the task that waits for either cancels the request, and the wait ends with
    /// `URLError.cancelled`.
    private final class ResponseChunks: NSObject, URLSessionDataDelegate, AsyncSequence, @unchecked Sendable {
        typealias Element = Data

        private struct State {
            var responseContinuation: CheckedContinuation<URLResponse, Error>?
            /// The chunks that arrived and haven't been read yet.
            var chunks: [Data] = []
            /// How the body ended, once it has: `nil` for its end, or the error it ended with.
            var ended: Result<Void, Error>?
            var chunkContinuation: CheckedContinuation<Data?, Error>?
            var task: URLSessionDataTask?
            /// The session made for this request alone, which ends with it.
            var ownedSession: URLSession?
            var isCancelled = false
        }

        // Read and written from the caller's task and from the delegate's queue.
        private let state = Locked(State())

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
                        continuation.resume(throwing: URLError(.cancelled))
                        return
                    }
                    task.resume()
                }
            } onCancel: {
                cancel()
            }
        }

        /// The next chunk of the body, or `nil` at its end.
        func next() async throws -> Data? {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data?, Error>) in
                    let ready: Result<Data?, Error>? = state.withLock { state in
                        if !state.chunks.isEmpty { return .success(state.chunks.removeFirst()) }
                        switch state.ended {
                        case .success?: return .success(nil)
                        case .failure(let error)?: return .failure(error)
                        case nil:
                            state.chunkContinuation = continuation
                            return nil
                        }
                    }
                    if let ready { continuation.resume(with: ready) }
                }
            } onCancel: {
                cancel()
            }
        }

        func makeAsyncIterator() -> Iterator { Iterator(chunks: self) }

        struct Iterator: AsyncIteratorProtocol {
            let chunks: ResponseChunks
            func next() async throws -> Data? { try await chunks.next() }
        }

        /// Cancels the request, and ends what waits for it with `URLError.cancelled`, as a URL
        /// session's own async calls do: FoundationNetworking doesn't always tell the delegate a
        /// cancelled task completed, so the wait isn't left to it.
        private func cancel() {
            let (task, response, waiting) = state.withLock { state in
                state.isCancelled = true
                if state.ended == nil { state.ended = .failure(URLError(.cancelled)) }
                return (state.task, Self.take(&state.responseContinuation), Self.take(&state.chunkContinuation))
            }
            response?.resume(throwing: URLError(.cancelled))
            waiting?.resume(throwing: URLError(.cancelled))
            task?.cancel()
        }

        /// Ends the request once it's been read: cancels it if it's still running.
        func end() {
            let (task, session) = state.withLock { state in
                defer {
                    state.task = nil
                    state.ownedSession = nil
                    if state.ended == nil { state.ended = .success(()) }
                }
                return (state.task, state.ownedSession)
            }
            task?.cancel()
            session?.invalidateAndCancel()
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
            let waiting = state.withLock { state -> CheckedContinuation<Data?, Error>? in
                guard state.ended == nil else { return nil }
                if let waiting = Self.take(&state.chunkContinuation) { return waiting }
                state.chunks.append(data)
                return nil
            }
            waiting?.resume(returning: data)
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
            let (response, waiting, owned) = state.withLock { state in
                defer { state.task = nil }
                if state.ended == nil { state.ended = error.map { .failure($0) } ?? .success(()) }
                return (
                    Self.take(&state.responseContinuation), Self.take(&state.chunkContinuation),
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
                waiting?.resume(throwing: error)
            } else {
                waiting?.resume(returning: nil)
            }
            owned?.finishTasksAndInvalidate()
        }
    }
#endif
