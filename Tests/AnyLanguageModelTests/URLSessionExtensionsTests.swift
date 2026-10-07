import Foundation
import Testing

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

@testable import AnyLanguageModel

@Suite("URLSession Extensions")
struct URLSessionExtensionsTests {
    @Test func invalidResponseDescriptionMatchesExpectedText() {
        let error = URLSessionError.invalidResponse
        #expect(error.description == "Invalid response")
    }

    @Test func httpErrorDescriptionIncludesStatusCodeAndDetail() {
        let error = URLSessionError.httpError(statusCode: 429, detail: "rate limit")
        #expect(error.description == "HTTP error (Status 429): rate limit")
    }

    @Test func decodingErrorDescriptionIncludesDetail() {
        let error = URLSessionError.decodingError(detail: "keyNotFound")
        #expect(error.description == "Decoding error: keyNotFound")
    }
}

#if canImport(FoundationNetworking)
    private actor GateCounter {
        private(set) var current = 0
        private(set) var maxConcurrent = 0

        func enter() {
            current += 1
            maxConcurrent = max(maxConcurrent, current)
        }

        func leave() {
            current -= 1
        }
    }

    private enum GateTestError: Error {
        case expected
    }

    private actor GateFlag {
        private(set) var value = false

        func setTrue() {
            value = true
        }
    }

    extension URLSessionExtensionsTests {
        @Test func linuxGateSerializesConcurrentOperations() async throws {
            let counter = GateCounter()

            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0 ..< 8 {
                    group.addTask {
                        try await withLinuxRequestLock {
                            await counter.enter()
                            do {
                                try await Task.sleep(for: .milliseconds(20))
                                await counter.leave()
                            } catch {
                                await counter.leave()
                                throw error
                            }
                        }
                    }
                }
                try await group.waitForAll()
            }

            #expect(await counter.maxConcurrent == 1)
        }

        @Test func linuxGateReleasesAfterError() async throws {
            do {
                try await withLinuxRequestLock {
                    throw GateTestError.expected
                }
                Issue.record("Expected error was not thrown")
            } catch GateTestError.expected {
                // expected
            }

            var ranSecondOperation = false
            try await withLinuxRequestLock {
                ranSecondOperation = true
            }
            #expect(ranSecondOperation)
        }

        @Test func linuxGateReleasesAfterCancellation() async throws {
            let longTask = Task {
                try await withLinuxRequestLock {
                    try await Task.sleep(for: .seconds(10))
                }
            }

            try await Task.sleep(for: .milliseconds(30))
            longTask.cancel()
            _ = await longTask.result

            var acquiredAfterCancellation = false
            try await withLinuxRequestLock {
                acquiredAfterCancellation = true
            }

            #expect(acquiredAfterCancellation)
        }

        @Test func linuxGateCancelledWaiterDoesNotExecute() async throws {
            let ranCancelledOperation = GateFlag()

            let holder = Task {
                try await withLinuxRequestLock {
                    try await Task.sleep(for: .milliseconds(200))
                }
            }

            try await Task.sleep(for: .milliseconds(20))

            let waiter = Task {
                do {
                    try await withLinuxRequestLock {
                        await ranCancelledOperation.setTrue()
                    }
                } catch {
                    // Cancellation is expected.
                }
            }

            waiter.cancel()
            _ = await waiter.result
            try await holder.value
            try await Task.sleep(for: .milliseconds(20))

            #expect(await ranCancelledOperation.value == false)
        }

        /// Canceling a stream before its response starts cancels the request
        /// and releases the gate, rather than holding it until a response arrives.
        @Test func linuxStreamCancelledBeforeResponseReleasesGate() async throws {
            struct Line: Decodable, Sendable {}

            for eventStream in [false, true] {
                SilentURLProtocol.reset()
                let session = SilentURLProtocol.makeSession()
                let url = URL(string: "https://example.com")!
                let streaming = Task {
                    let stream: AsyncThrowingStream<Line, any Error> =
                        eventStream
                        ? session.fetchEventStream(HTTP.Method.post, url: url)
                        : session.fetchStream(HTTP.Method.post, url: url)
                    for try await _ in stream {}
                }

                let deadline = ContinuousClock.now + .seconds(5)
                while !SilentURLProtocol.didStartLoading, ContinuousClock.now < deadline {
                    try await Task.sleep(for: .milliseconds(10))
                }
                #expect(SilentURLProtocol.didStartLoading)

                streaming.cancel()
                let acquired = try await withThrowingTaskGroup(of: Bool.self) { group in
                    group.addTask {
                        try await withLinuxRequestLock {}
                        return true
                    }
                    group.addTask {
                        try await Task.sleep(for: .seconds(2))
                        return false
                    }
                    let first = try await group.next() ?? false
                    group.cancelAll()
                    return first
                }
                #expect(acquired, "fetchEventStream: \(eventStream)")

                // If the request is still waiting, fail it so it releases the gate for other tests.
                SilentURLProtocol.failPendingRequests()
                _ = await streaming.result
            }
        }
    }

    /// Starts loading and never responds, like a server that hasn't sent headers yet.
    private final class SilentURLProtocol: URLProtocol, @unchecked Sendable {
        // URLProtocol isn't Sendable on Linux; every access goes through `state`'s lock.
        private struct State: @unchecked Sendable {
            var didStartLoading = false
            var pending: [SilentURLProtocol] = []
        }

        private static let state = Locked(State())

        static var didStartLoading: Bool { state.withLock { $0.didStartLoading } }

        static func reset() {
            state.withLock { $0 = State() }
        }

        static func failPendingRequests() {
            let pending = state.withLock { state in
                defer { state.pending = [] }
                return state.pending
            }
            for request in pending {
                request.client?.urlProtocol(request, didFailWithError: URLError(.timedOut))
            }
        }

        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [SilentURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            Self.state.withLock { state in
                state.didStartLoading = true
                state.pending.append(self)
            }
        }

        override func stopLoading() {
            Self.state.withLock { state in
                state.pending.removeAll { $0 === self }
            }
        }
    }
#endif

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension URLSessionExtensionsTests {
        private struct Line: Decodable, Sendable, Equatable {
            let n: Int
        }

        /// Each line arrives while the response is still loading, not when it ends.
        @Test func jsonLinesStreamAsTheyArrive() async throws {
            let session = ChunkedURLProtocol.makeSession()
            var received: [Line] = []
            let stream: AsyncThrowingStream<Line, any Error> = session.fetchStream(
                .post,
                url: URL(string: "https://example.com")!
            )
            for try await line in stream {
                received.append(line)
                // The second line is sent only once the first has been received.
                if line.n == 1 { ChunkedURLProtocol.firstLineReceived.signal() }
            }
            #expect(received == [Line(n: 1), Line(n: 2), Line(n: 3)])
            #expect(ChunkedURLProtocol.sentRestAfterFirstLine)
        }
    }

    /// Sends `{"n":1}` and a newline, waits until the test has received it, then sends the rest,
    /// with a last line that has no newline.
    private final class ChunkedURLProtocol: URLProtocol, @unchecked Sendable {
        static let firstLineReceived = DispatchSemaphore(value: 0)
        nonisolated(unsafe) static var sentRestAfterFirstLine = false

        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [ChunkedURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/x-ndjson"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(#"{"n":1}"#.utf8 + [UInt8(ascii: "\n")]))
            DispatchQueue.global().async {
                Self.sentRestAfterFirstLine = Self.firstLineReceived.wait(timeout: .now() + 5) == .success
                self.client?.urlProtocol(self, didLoad: Data("{\"n\":2}\n{\"n\":3}".utf8))
                self.client?.urlProtocolDidFinishLoading(self)
            }
        }

        override func stopLoading() {}
    }
#endif
