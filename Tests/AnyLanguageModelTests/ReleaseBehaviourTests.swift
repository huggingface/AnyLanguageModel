import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

#if !canImport(AsyncHTTPClient)
    extension URLSessionExtensionsTests {
        /// What 1.0 does with a URL session a model is given, through public API only, so the same
        /// tests run against 1.0 and against what changes it.
        @Suite("Release behaviour with a URL session", .timeLimit(.minutes(1)))
        struct ReleaseBehaviour {
            @available(*, deprecated)
            private static func model(_ server: LoopbackServer, sessionTimeout: TimeInterval = 30)
                -> OllamaLanguageModel
            {
                let configuration = URLSessionConfiguration.ephemeral
                configuration.timeoutIntervalForRequest = sessionTimeout
                return OllamaLanguageModel(
                    baseURL: server.url,
                    model: "m",
                    session: URLSession(configuration: configuration)
                )
            }

            /// The session's own timeout is how long a request waits, shorter or longer.
            @available(*, deprecated)
            @Test func aSessionsTimeoutBoundsTheWaitForAResponse() async throws {
                let server = try LoopbackServer(.silence)
                defer { server.stop() }
                let started = Date()
                let error = await #expect(throws: URLError.self) {
                    try await LanguageModelSession(model: Self.model(server, sessionTimeout: 1)).respond(to: "Hi")
                }
                #expect(error?.code == .timedOut)
                #expect(Date().timeIntervalSince(started) < 3)
            }

            /// What a response throws when it's cancelled after `delay`.
            private static func cancelled(_ model: OllamaLanguageModel, after delay: Duration) async -> (any Error)? {
                let task = Task { _ = try await LanguageModelSession(model: model).respond(to: "Hi") }
                try? await Task.sleep(for: delay)
                task.cancel()
                do {
                    try await task.value
                    return nil
                } catch {
                    return error
                }
            }

            @available(*, deprecated)
            @Test func cancellingBeforeTheResponseThrowsURLErrorCancelled() async throws {
                let server = try LoopbackServer(.silence)
                defer { server.stop() }
                let error = await Self.cancelled(Self.model(server), after: .milliseconds(300))
                #expect((error as? URLError)?.code == .cancelled, "\(String(describing: error))")
            }

            @available(*, deprecated)
            @Test func cancellingWhileTheBodyArrivesThrowsURLErrorCancelled() async throws {
                let server = try LoopbackServer(.partial(contentLength: 1000, body: #"{"model":"#))
                defer { server.stop() }
                let error = await Self.cancelled(Self.model(server), after: .milliseconds(500))
                #expect((error as? URLError)?.code == .cancelled, "\(String(describing: error))")
            }
        }
    }
#endif
