import Foundation
import Testing

// This file imports AnyLanguageModel without `@testable`,
// so it checks that callers outside the module can match these errors.
import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Provider errors", .serialized)
    struct ProviderErrorTests {
        @Generable
        struct Answer {
            let value: String
        }

        @Test func openAIChatCompletionsWithoutChoices() async throws {
            CannedURLProtocol.respond(with: #"{"id": "test", "choices": []}"#)
            let model = OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                apiVariant: .chatCompletions,
                session: CannedURLProtocol.makeSession()
            )
            let session = LanguageModelSession(model: model)

            do {
                _ = try await session.respond(to: "Hello")
                Issue.record("Expected OpenAILanguageModelError.noResponseGenerated")
            } catch OpenAILanguageModelError.noResponseGenerated {
                // Expected.
            }
        }

        @Test func openAIResponsesWithoutStructuredOutput() async throws {
            CannedURLProtocol.respond(with: #"{"id": "test", "output": []}"#)
            let model = OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                apiVariant: .responses,
                session: CannedURLProtocol.makeSession()
            )
            let session = LanguageModelSession(model: model)

            await #expect(throws: OpenAILanguageModelError.self) {
                _ = try await session.respond(to: "Hello", generating: Answer.self)
            }
        }

        @Test func openResponsesWithoutStructuredOutput() async throws {
            CannedURLProtocol.respond(with: #"{"id": "test", "output": []}"#)
            let session = LanguageModelSession(model: Self.openResponsesModel())

            do {
                _ = try await session.respond(to: "Hello", generating: Answer.self)
                Issue.record("Expected OpenResponsesLanguageModelError.noResponseGenerated")
            } catch let error as OpenResponsesLanguageModelError {
                guard case .noResponseGenerated = error else {
                    Issue.record("Unexpected error: \(error)")
                    return
                }
                #expect(error.errorDescription != nil)
            }
        }

        /// A `response.failed` event as the OpenAI Responses API and Open Responses send it.
        private static let failedEvent = """
            data: {"type": "response.failed", "sequence_number": 1, "response": {"id": "test", \
            "status": "failed", "error": {"code": "server_error", "message": "The model failed."}}}


            """

        @Test func openAIResponsesStreamFailure() async throws {
            CannedURLProtocol.respond(with: Self.failedEvent)
            let model = OpenAILanguageModel(
                apiKey: "test",
                model: "test",
                apiVariant: .responses,
                session: CannedURLProtocol.makeSession()
            )
            let session = LanguageModelSession(model: model)

            do {
                for try await _ in session.streamResponse(to: "Hello") {}
                Issue.record("Expected OpenAILanguageModelError.streamFailed")
            } catch OpenAILanguageModelError.streamFailed(let code, let message) {
                #expect(code == "server_error")
                #expect(message == "The model failed.")
            }
        }

        @Test func openResponsesStreamFailure() async throws {
            CannedURLProtocol.respond(with: Self.failedEvent)
            let session = LanguageModelSession(model: Self.openResponsesModel())

            do {
                for try await _ in session.streamResponse(to: "Hello") {}
                Issue.record("Expected OpenResponsesLanguageModelError.streamFailed")
            } catch let error as OpenResponsesLanguageModelError {
                guard case .streamFailed(let code, let message) = error else {
                    Issue.record("Unexpected error: \(error)")
                    return
                }
                #expect(code == "server_error")
                #expect(message == "The model failed.")
                #expect(
                    error.errorDescription == "The response failed while streaming (server_error): The model failed."
                )
            }
        }

        @Test func openResponsesStreamFailureWithoutDetails() async throws {
            CannedURLProtocol.respond(with: "data: {\"type\": \"response.failed\"}\n\n")
            let session = LanguageModelSession(model: Self.openResponsesModel())

            do {
                for try await _ in session.streamResponse(to: "Hello") {}
                Issue.record("Expected OpenResponsesLanguageModelError.streamFailed")
            } catch OpenResponsesLanguageModelError.streamFailed(let code, let message) {
                #expect(code == nil)
                #expect(message == nil)
            }
        }

        private static func openResponsesModel() -> OpenResponsesLanguageModel {
            OpenResponsesLanguageModel(
                baseURL: URL(string: "https://example.com/v1")!,
                apiKey: "test",
                model: "test",
                session: CannedURLProtocol.makeSession()
            )
        }
    }

    /// A `URLProtocol` that answers every request with one canned body.
    private final class CannedURLProtocol: URLProtocol {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var body = Data()

        /// Sets the body returned to every request.
        static func respond(with body: String) {
            lock.withLock { self.body = Data(body.utf8) }
        }

        /// A session that routes every request to this protocol.
        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [CannedURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            guard let url = request.url else {
                client?.urlProtocol(self, didFailWithError: URLError(.badURL))
                return
            }
            let isStream = request.value(forHTTPHeaderField: "Accept") == "text/event-stream"
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": isStream ? "text/event-stream" : "application/json"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Self.lock.withLock { Self.body })
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }
#endif
