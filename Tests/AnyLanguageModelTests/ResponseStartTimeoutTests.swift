import Foundation
import HTTPTypes
import HTTPTypesFoundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
#endif

/// The transports' requests run one at a time with the other tests that send through a URL session,
/// as on Linux they share one request gate.
extension URLSessionExtensionsTests {
    @Suite("Response start timeout", .timeLimit(.minutes(1)))
    struct ResponseStartTimeout {
        enum Provider: CaseIterable {
            case chat, responses, openResponses, anthropic, gemini, ollama

            func model(responseStartTimeout: Duration?, transport: any HTTPTransport) -> any LanguageModel {
                switch self {
                case .chat, .responses:
                    OpenAILanguageModel(
                        apiKey: "test",
                        model: "test",
                        apiVariant: self == .chat ? .chatCompletions : .responses,
                        responseStartTimeout: responseStartTimeout,
                        transport: transport
                    )
                case .openResponses:
                    OpenResponsesLanguageModel(
                        baseURL: URL(string: "https://example.com/v1")!,
                        apiKey: "test",
                        model: "test",
                        responseStartTimeout: responseStartTimeout,
                        transport: transport
                    )
                case .anthropic:
                    AnthropicLanguageModel(
                        apiKey: "test",
                        model: "test",
                        responseStartTimeout: responseStartTimeout,
                        transport: transport
                    )
                case .gemini:
                    GeminiLanguageModel(
                        apiKey: "test",
                        model: "test",
                        responseStartTimeout: responseStartTimeout,
                        transport: transport
                    )
                case .ollama:
                    OllamaLanguageModel(model: "test", responseStartTimeout: responseStartTimeout, transport: transport)
                }
            }
        }

        /// Every provider sends a streamed request with its model's timeout, or with none for the
        /// transport to decide, whichever transport sends it.
        @Test(arguments: Provider.allCases, [Duration.seconds(7), nil])
        func streamedRequestsCarryTheModelsTimeout(_ provider: Provider, _ timeout: Duration?) async throws {
            let transport = RecordingTransport()
            let model = provider.model(responseStartTimeout: timeout, transport: transport)
            _ = try? await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()
            #expect(await transport.timeouts.recorded == [timeout])
        }

        /// The timeout is for a streamed response to start: a whole response is the transport's to wait for.
        @Test(arguments: Provider.allCases)
        func wholeResponsesCarryNoTimeout(_ provider: Provider) async throws {
            let transport = RecordingTransport()
            let model = provider.model(responseStartTimeout: .seconds(7), transport: transport)
            _ = try? await LanguageModelSession(model: model).respond(to: "Hi")
            #expect(await transport.timeouts.recorded == [nil])
        }

        /// Each transport stops waiting for a response that doesn't start at the timeout it's given,
        /// not at a default and not in other units. A URL session on Linux waits whole seconds.
        @Test(arguments: Transport.available, [0.5, 1.5])
        func transportStopsWaitingAtTheTimeout(_ transport: Transport, _ seconds: Double) async throws {
            let server = try LoopbackServer(.silence)
            defer { server.stop() }
            var expected = seconds
            #if canImport(FoundationNetworking)
                if transport.name == "URLSession" { expected = max(1, seconds.rounded(.up)) }
            #endif
            let start = ContinuousClock.now
            await #expect(throws: (any Error).self) {
                try await transport.transport.send(
                    request: HTTPRequest(url: server.url),
                    body: nil,
                    timeout: .milliseconds(Int(seconds * 1000))
                ) { _, _ in }
            }
            let waited = ContinuousClock.now - start
            #expect(waited >= .milliseconds(Int(expected * 1000) - 100), "\(waited)")
            #expect(waited < .milliseconds(Int(expected * 1000) + 1500), "\(waited)")
        }

        /// A streamed response's start timeout too long to wait for at all is no limit to speak of
        /// through a URL session, rather than a trap on Linux.
        @Test func aHugeResponseStartTimeoutDoesntTrap() async throws {
            let server = try LoopbackServer(.partial(contentLength: 2, body: "{}"))
            defer { server.stop() }
            let model = OllamaLanguageModel(
                baseURL: server.url, model: "test", responseStartTimeout: .seconds(Int64.max),
                transport: URLSession(configuration: .ephemeral))
            // The body isn't an answer, so it fails, but as a decoding error, not a crash.
            await #expect(throws: (any Error).self) {
                try await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()
            }
        }

        /// A transport the timeout test sends with.
        struct Transport: Sendable, CustomTestStringConvertible {
            let name: String
            let transport: any HTTPTransport

            var testDescription: String { name }

            static var available: [Transport] {
                var transports = [Transport(name: "URLSession", transport: URLSession(configuration: .ephemeral))]
                #if canImport(AsyncHTTPClient)
                    transports.append(Transport(name: "HTTPClient", transport: HTTPClient.shared))
                #endif
                return transports
            }
        }
    }
}

/// Records the timeout each request is sent with, and answers none.
private struct RecordingTransport: HTTPTransport {
    struct Unanswered: Error {}

    actor Timeouts {
        private(set) var recorded: [Duration?] = []

        func record(_ timeout: Duration?) {
            recorded.append(timeout)
        }
    }

    let timeouts = Timeouts()

    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        await timeouts.record(timeout)
        throw Unanswered()
    }
}
