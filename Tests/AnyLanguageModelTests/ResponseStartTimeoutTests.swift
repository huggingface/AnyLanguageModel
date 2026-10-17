import Foundation
import HTTPTypes
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
#endif
#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

@Suite("Response start timeout")
struct ResponseStartTimeoutTests {
    enum Provider: CaseIterable {
        case chat, responses, openResponses, anthropic, gemini, ollama

        func model(responseStartTimeout: Duration, session: any HTTPTransport) -> any LanguageModel {
            switch self {
            case .chat, .responses:
                OpenAILanguageModel(
                    apiKey: "test",
                    model: "test",
                    apiVariant: self == .chat ? .chatCompletions : .responses,
                    responseStartTimeout: responseStartTimeout,
                    session: session
                )
            case .openResponses:
                OpenResponsesLanguageModel(
                    baseURL: URL(string: "https://example.com/v1")!,
                    apiKey: "test",
                    model: "test",
                    responseStartTimeout: responseStartTimeout,
                    session: session
                )
            case .anthropic:
                AnthropicLanguageModel(
                    apiKey: "test",
                    model: "test",
                    responseStartTimeout: responseStartTimeout,
                    session: session
                )
            case .gemini:
                GeminiLanguageModel(
                    apiKey: "test",
                    model: "test",
                    responseStartTimeout: responseStartTimeout,
                    session: session
                )
            case .ollama:
                OllamaLanguageModel(model: "test", responseStartTimeout: responseStartTimeout, session: session)
            }
        }
    }

    /// Every provider sends a streamed request with its model's timeout, whichever transport sends it.
    @Test(arguments: Provider.allCases)
    func streamedRequestsCarryTheModelsTimeout(_ provider: Provider) async throws {
        let transport = RecordingTransport()
        let model = provider.model(responseStartTimeout: .seconds(7), session: transport)
        _ = try? await LanguageModelSession(model: model).streamResponse(to: "Hi").collect()
        #expect(await transport.timeouts.recorded == [.seconds(7)])
    }

    /// Each transport stops waiting for a response that doesn't start at the timeout it's given,
    /// not at a default and not in other units.
    @Test(.timeLimit(.minutes(1)), arguments: Transport.available)
    func transportStopsWaitingAtTheTimeout(_ transport: Transport) async throws {
        let server = try SilentServer()
        defer { server.close() }
        let request = HTTPRequest(method: .get, scheme: "http", authority: "127.0.0.1:\(server.port)", path: "/")
        let start = ContinuousClock.now
        await #expect(throws: (any Error).self) {
            try await transport.session.perform(request: request, body: nil, timeout: .milliseconds(500)) { _, _ in }
        }
        let waited = ContinuousClock.now - start
        #expect(waited >= .milliseconds(400))
        #expect(waited < .seconds(10))
    }

    /// A transport the timeout test sends with.
    struct Transport: Sendable, CustomTestStringConvertible {
        let name: String
        let session: any HTTPTransport

        var testDescription: String { name }

        static var available: [Transport] {
            var transports = [Transport(name: "URLSession", session: URLSession(configuration: .ephemeral))]
            #if canImport(AsyncHTTPClient)
                transports.append(Transport(name: "HTTPClient", session: HTTPClient.shared))
            #endif
            return transports
        }
    }
}

/// Records the timeout each request is sent with, and answers none.
private struct RecordingTransport: HTTPTransport {
    struct Unanswered: Error {}

    actor Timeouts {
        private(set) var recorded: [Duration] = []

        func record(_ timeout: Duration) {
            recorded.append(timeout)
        }
    }

    let timeouts = Timeouts()

    func perform<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        await timeouts.record(timeout)
        throw Unanswered()
    }
}

/// A local server that takes connections and never answers: the system accepts them
/// into its backlog, and nothing reads them.
private struct SilentServer {
    let descriptor: Int32
    let port: UInt16

    struct Failed: Error {
        let call: String
    }

    init() throws {
        #if canImport(Darwin)
            let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        #else
            let descriptor = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #endif
        guard descriptor >= 0 else { throw Failed(call: "socket") }
        var address = sockaddr_in()
        #if canImport(Darwin)
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        #endif
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { address in
                bind(descriptor, address, length) == 0 && listen(descriptor, 8) == 0
                    && getsockname(descriptor, address, &length) == 0
            }
        }
        guard bound else {
            _ = closeDescriptor(descriptor)
            throw Failed(call: "bind")
        }
        self.descriptor = descriptor
        self.port = UInt16(bigEndian: address.sin_port)
    }

    func close() {
        _ = closeDescriptor(descriptor)
    }
}

/// The system's `close`, which `SilentServer.close()` hides inside the server.
private func closeDescriptor(_ descriptor: Int32) -> Int32 {
    close(descriptor)
}
