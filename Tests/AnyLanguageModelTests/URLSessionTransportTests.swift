import Foundation
import HTTPTypes
import HTTPTypesFoundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

extension URLSessionExtensionsTests {
    /// How a URL session sends a request: how long it waits, and what a cancelled one throws.
    @Suite("URL session transport", .timeLimit(.minutes(1)))
    struct Transport {
        /// Sends a request to `server` with `timeout`, through a session that waits
        /// `sessionTimeout`: what it throws, and after how long.
        private static func failure(
            _ server: LoopbackServer,
            timeout: Duration?,
            sessionTimeout: TimeInterval
        ) async -> (URLError?, Double) {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForRequest = sessionTimeout
            let session = URLSession(configuration: configuration)
            let started = Date()
            do {
                _ = try await session.send(
                    request: HTTPRequest(url: server.url),
                    body: nil,
                    timeout: timeout
                ) { _, body in try await body.collect() }
                return (nil, Date().timeIntervalSince(started))
            } catch {
                return (error as? URLError, Date().timeIntervalSince(started))
            }
        }

        @Test func aGivenTimeoutIsUsedWhenTheSessionWaitsLonger() async throws {
            let server = try LoopbackServer(.silence)
            defer { server.stop() }
            let (error, seconds) = await Self.failure(server, timeout: .seconds(1), sessionTimeout: 3)
            #expect(error?.code == .timedOut)
            #expect(seconds >= 0.8 && seconds < 2.5, "\(seconds)")
        }

        @Test func aGivenTimeoutIsUsedWhenTheSessionWaitsLess() async throws {
            let server = try LoopbackServer(.silence)
            defer { server.stop() }
            let (error, seconds) = await Self.failure(server, timeout: .seconds(2), sessionTimeout: 1)
            #expect(error?.code == .timedOut)
            #expect(seconds >= 1.8 && seconds < 3.5, "\(seconds)")
        }

        @Test func withoutATimeoutTheSessionsOwnApplies() async throws {
            let server = try LoopbackServer(.silence)
            defer { server.stop() }
            let (error, seconds) = await Self.failure(server, timeout: nil, sessionTimeout: 1)
            #expect(error?.code == .timedOut)
            #expect(seconds >= 0.8 && seconds < 2.5, "\(seconds)")
        }

        @Test func cancellingWhileTheBodyIsReadThrowsURLErrorCancelled() async throws {
            let server = try LoopbackServer(.partial(contentLength: 1000, body: "{"))
            defer { server.stop() }
            let session = URLSession(configuration: .ephemeral)
            let task = Task {
                try await session.send(
                    request: HTTPRequest(url: server.url),
                    body: nil,
                    timeout: .seconds(30)
                ) { _, body in try await body.collect() }
            }
            try await Task.sleep(for: .milliseconds(500))
            task.cancel()
            let error = await #expect(throws: URLError.self) { try await task.value }
            #expect(error?.code == .cancelled)
        }

        /// A timeout too long for FoundationNetworking to count in milliseconds is no limit to
        /// speak of, rather than a trap.
        @Test func aHugeTimeoutDoesntTrap() async throws {
            let server = try LoopbackServer(.partial(contentLength: 2, body: "{}"))
            defer { server.stop() }
            let body = try await URLSession(configuration: .ephemeral).send(
                request: HTTPRequest(url: server.url),
                body: nil,
                timeout: .seconds(Int64.max)
            ) { _, body in try await body.collect() }
            #expect(body == Data("{}".utf8))
        }

        @Test func aWholeBodyIsRead() async throws {
            let server = try LoopbackServer(.partial(contentLength: 2, body: "{}"))
            defer { server.stop() }
            let body = try await URLSession(configuration: .ephemeral).send(
                request: HTTPRequest(url: server.url),
                body: nil,
                timeout: .seconds(5)
            ) { _, body in try await body.collect() }
            #expect(body == Data("{}".utf8))
        }
    }
}
