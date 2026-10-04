import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Ollama request history", .serialized)
    struct OllamaRequestHistoryTests {
        private func response(_ content: String) -> String {
            """
            {"model": "test", "created_at": "2026-10-03T00:00:00.000Z",
             "message": {"role": "assistant", "content": "\(content)"}, "done": true}
            """
        }

        @Test func nonstreamingRequestsSendInstructionsAndHistory() async throws {
            OllamaHistoryURLProtocol.reset(responses: [response("Hi"), response("Hi again")])

            let model = OllamaLanguageModel(model: "test", session: OllamaHistoryURLProtocol.makeSession())
            let session = LanguageModelSession(model: model, instructions: "Be brief.")
            _ = try await session.respond(to: "Hello")
            _ = try await session.respond(to: "Again")

            let body = try #require(OllamaHistoryURLProtocol.recordedBodies.last)
            let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            let messages = try #require(json["messages"] as? [[String: Any]])
            #expect(messages.map { $0["role"] as? String } == ["system", "user", "assistant", "user"])
            #expect(messages.map { $0["content"] as? String } == ["Be brief.", "Hello", "Hi", "Again"])
        }
    }

    /// A `URLProtocol` with its own queue of JSON responses,
    /// so this suite doesn't share state with suites that run in parallel.
    private final class OllamaHistoryURLProtocol: URLProtocol {
        private struct State: Sendable {
            var pending: [String] = []
            var recordedBodies: [Data] = []
        }

        private static let state = Locked(State())

        static func reset(responses: [String]) {
            state.withLock { $0 = State(pending: responses) }
        }

        static var recordedBodies: [Data] {
            state.withLock { $0.recordedBodies }
        }

        static func makeSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [OllamaHistoryURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            // URLSession moves `httpBody` to `httpBodyStream` before the protocol sees the request.
            let body = request.httpBody ?? request.httpBodyStream.map(Self.readAll) ?? Data()
            let next = Self.state.withLock { state -> String? in
                state.recordedBodies.append(body)
                return state.pending.isEmpty ? nil : state.pending.removeFirst()
            }
            guard let next, let url = request.url else {
                client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
                return
            }
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(next.utf8))
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}

        private static func readAll(_ stream: InputStream) -> Data {
            stream.open()
            defer { stream.close() }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while true {
                let read = stream.read(&buffer, maxLength: buffer.count)
                if read <= 0 { break }
                data.append(buffer, count: read)
            }
            return data
        }
    }
#endif
