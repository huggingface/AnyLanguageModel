import EventSource
import Foundation
import HTTPTypes
import HTTPTypesFoundation

/// Why a request to a model's provider failed in sending it or in its answer.
///
/// A caller can read an error response's status, headers and body,
/// for example to wait as its `Retry-After` header says.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public enum HTTPTransportError: Error, CustomStringConvertible {
    /// The request couldn't be sent as given.
    case invalidRequest
    /// The answer wasn't an HTTP response.
    case invalidResponse
    /// The provider answered with an error: its response, headers included, and its body.
    case httpError(response: HTTPResponse, body: String)
    /// A successful answer couldn't be read as what was expected.
    case decodingError(detail: String)

    public var description: String {
        switch self {
        case .invalidRequest:
            return "Invalid request"
        case .invalidResponse:
            return "Invalid response"
        case .httpError(let response, let body):
            return "HTTP error (Status \(response.status.code)): \(body)"
        case .decodingError(let detail):
            return "Decoding error: \(detail)"
        }
    }
}

/// What every session's requests to a model's provider are made of, written once on top of
/// ``HTTPTransport/perform(request:body:timeout:responseHandler:)``: an answer decoded from JSON,
/// one decoded from each line of a stream as it arrives, and one from each event of a stream
/// of server-sent events.
extension HTTPTransport {
    func fetch<T: Decodable>(
        _ request: HTTPRequest,
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) async throws -> T {
        let request = Self.request(request, accepting: "application/json", body: body)
        let data = try await perform(request: request, body: body, timeout: .seconds(180)) { response, body in
            try await Self.validate(response, body)
            return try await body.collect()
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = dateDecodingStrategy
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw HTTPTransportError.decodingError(detail: error.localizedDescription)
        }
    }

    func fetchStream<T: Decodable & Sendable>(
        _ request: HTTPRequest,
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate,
        timeout: Duration = .seconds(60)
    ) -> AsyncThrowingStream<T, any Error> {
        let request = Self.request(request, accepting: "application/json", body: body)
        return streaming { continuation in
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = dateDecodingStrategy
            try await perform(request: request, body: body, timeout: timeout) { response, body in
                try await Self.validate(response, body)
                // Each line is decoded as it arrives.
                var lines = JSONLines()
                for try await byte in body {
                    guard let line = lines.append(byte) else { continue }
                    try Task.checkCancellation()
                    continuation.yield(try decoder.decode(T.self, from: Data(line)))
                }
                if let line = lines.finish() {
                    continuation.yield(try decoder.decode(T.self, from: Data(line)))
                }
            }
        }
    }

    func fetchEventStream<T: Decodable & Sendable>(
        _ request: HTTPRequest,
        body: Data? = nil,
        timeout: Duration = .seconds(60)
    ) -> AsyncThrowingStream<T, any Error> {
        let request = Self.request(request, accepting: "text/event-stream", body: body)
        return streaming { continuation in
            try await perform(request: request, body: body, timeout: timeout) { response, body in
                try await Self.validate(response, body)
                let decoder = JSONDecoder()
                for try await event in body.events {
                    guard let data = event.data.data(using: .utf8) else { continue }
                    if let decoded = try? decoder.decode(T.self, from: data) {
                        continuation.yield(decoded)
                    }
                }
            }
        }
    }

    /// A stream of what `produce` yields, in a task of its own that ends with it:
    /// ending the stream early cancels the task, and with it the request.
    private func streaming<T: Sendable>(
        _ produce: @escaping @Sendable (AsyncThrowingStream<T, any Error>.Continuation) async throws -> Void
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await produce(continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// `request`, accepting `accept` and saying it sends JSON when it has `body`, unless it says
    /// otherwise.
    private static func request(_ request: HTTPRequest, accepting accept: String, body: Data?) -> HTTPRequest {
        var request = request
        if request.headerFields[.accept] == nil { request.headerFields[.accept] = accept }
        if body != nil, request.headerFields[.contentType] == nil { request.headerFields[.contentType] = "application/json" }
        return request
    }

    /// A response that isn't a success fails with what its body says.
    private static func validate(_ response: HTTPResponse, _ body: HTTPResponseBody) async throws {
        guard response.status.kind == .successful else {
            let detail = String(data: try await body.collect(), encoding: .utf8) ?? "Invalid response"
            throw HTTPTransportError.httpError(response: response, body: detail)
        }
    }
}
