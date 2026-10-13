import EventSource
import Foundation
import JSONSchema

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

enum HTTP {
    enum Method: String {
        case get = "GET"
        case post = "POST"
    }
}

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

extension URLSession {
    func fetch<T: Decodable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.addValue("application/json", forHTTPHeaderField: "Accept")

        for (key, value) in headers {
            request.addValue(value, forHTTPHeaderField: key)
        }

        if let body {
            request.httpBody = body
            request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        #if canImport(FoundationNetworking)
            var lockedData: Data?
            var lockedResponse: URLResponse?
            try await withLinuxRequestLock {
                let (data, response) = try await data(for: request)
                lockedData = data
                lockedResponse = response
            }
            guard let data = lockedData, let response = lockedResponse else {
                throw URLSessionError.invalidResponse
            }
        #else
            let (data, response) = try await data(for: request)
        #endif

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLSessionError.invalidResponse
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = dateDecodingStrategy

        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            if let errorString = String(data: data, encoding: .utf8) {
                throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: errorString)
            }
            throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: "Invalid response")
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw URLSessionError.decodingError(detail: error.localizedDescription)
        }
    }

    func fetchStream<T: Decodable & Sendable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = dateDecodingStrategy

                do {
                    var request = URLRequest(url: url)
                    request.httpMethod = method.rawValue
                    request.addValue("application/json", forHTTPHeaderField: "Accept")

                    for (key, value) in headers {
                        request.addValue(value, forHTTPHeaderField: key)
                    }

                    if let body {
                        request.httpBody = body
                        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
                    }

                    #if canImport(FoundationNetworking)
                        var lockedAsyncBytes: AsyncThrowingStream<UInt8, Error>?
                        var lockedResponse: URLResponse?
                        try await withLinuxRequestLock {
                            let (bytes, response) = try await self.linuxBytes(for: request)
                            lockedAsyncBytes = bytes
                            lockedResponse = response
                        }
                        guard let asyncBytes = lockedAsyncBytes, let response = lockedResponse else {
                            throw URLSessionError.invalidResponse
                        }
                        try await self.validateStreamingResponse(response, asyncBytes: asyncBytes)
                        try await decodeAndYieldJSONLines(asyncBytes, using: decoder, to: continuation)
                    #else
                        let (asyncBytes, response) = try await self.bytes(for: request)
                        try await validateStreamingResponse(response, asyncBytes: asyncBytes)
                        try await decodeAndYieldJSONLines(asyncBytes, using: decoder, to: continuation)
                    #endif
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    func fetchEventStream<T: Decodable & Sendable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    var request = URLRequest(url: url)
                    request.httpMethod = method.rawValue
                    request.addValue("text/event-stream", forHTTPHeaderField: "Accept")

                    for (key, value) in headers {
                        request.addValue(value, forHTTPHeaderField: key)
                    }

                    if let body {
                        request.httpBody = body
                        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
                    }

                    #if canImport(FoundationNetworking)
                        var lockedAsyncBytes: AsyncThrowingStream<UInt8, Error>?
                        var lockedResponse: URLResponse?
                        try await withLinuxRequestLock {
                            let (bytes, response) = try await self.linuxBytes(for: request)
                            lockedAsyncBytes = bytes
                            lockedResponse = response
                        }
                        guard let asyncBytes = lockedAsyncBytes, let response = lockedResponse else {
                            throw URLSessionError.invalidResponse
                        }
                        try await self.validateStreamingResponse(response, asyncBytes: asyncBytes)
                        try await decodeAndYieldEventStream(asyncBytes, to: continuation)
                    #else
                        let (asyncBytes, response) = try await self.bytes(for: request)
                        try await validateStreamingResponse(response, asyncBytes: asyncBytes)
                        try await decodeAndYieldEventStream(asyncBytes, to: continuation)
                    #endif
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func validateStreamingResponse<Bytes>(
        _ response: URLResponse,
        asyncBytes: Bytes
    ) async throws where Bytes: AsyncSequence, Bytes.Element == UInt8 {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLSessionError.invalidResponse
        }

        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            var errorData = Data()
            for try await byte in asyncBytes {
                errorData.append(byte)
            }
            if let errorString = String(data: errorData, encoding: .utf8) {
                throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: errorString)
            }
            throw URLSessionError.httpError(statusCode: httpResponse.statusCode, detail: "Invalid response")
        }
    }

    /// Decodes each line as it arrives, the way the AsyncHTTPClient transport does.
    private func decodeAndYieldJSONLines<T: Decodable & Sendable, Bytes>(
        _ asyncBytes: Bytes,
        using decoder: JSONDecoder,
        to continuation: AsyncThrowingStream<T, any Error>.Continuation
    ) async throws where Bytes: AsyncSequence, Bytes.Element == UInt8 {
        var lines = JSONLines()
        for try await byte in asyncBytes {
            guard let line = lines.append(byte) else { continue }
            try Task.checkCancellation()
            continuation.yield(try decoder.decode(T.self, from: Data(line)))
        }
        if let line = lines.finish() {
            continuation.yield(try decoder.decode(T.self, from: Data(line)))
        }
    }

    private func decodeAndYieldEventStream<T: Decodable & Sendable, Bytes>(
        _ asyncBytes: Bytes,
        to continuation: AsyncThrowingStream<T, any Error>.Continuation
    ) async throws where Bytes: AsyncSequence, Bytes.Element == UInt8 {
        let decoder = JSONDecoder()
        for try await event in asyncBytes.events {
            guard let data = event.data.data(using: .utf8) else { continue }
            if let decoded = try? decoder.decode(T.self, from: data) {
                continuation.yield(decoded)
            }
        }
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
                    continuation.resume(throwing: URLSessionError.invalidResponse)
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

enum URLSessionError: Error, CustomStringConvertible {
    case invalidResponse
    case httpError(statusCode: Int, detail: String)
    case decodingError(detail: String)

    var description: String {
        switch self {
        case .invalidResponse:
            return "Invalid response"
        case .httpError(let statusCode, let detail):
            return "HTTP error (Status \(statusCode)): \(detail)"
        case .decodingError(let detail):
            return "Decoding error: \(detail)"
        }
    }
}
