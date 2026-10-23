import EventSource
import Foundation
import HTTPTypes
#if !os(WASI)
    import HTTPTypesFoundation
#endif

/// Why a request to a model's provider failed: it couldn't be made, its answer wasn't an HTTP
/// response, the provider answered with an error status, or a successful answer couldn't be read.
///
/// A caller can read an error response's status, headers and body,
/// for example to wait as its `Retry-After` header says.
///
/// A request that fails on the way, such as one the network drops, throws the transport's own
/// error instead: `URLError` from a URL session, or `HTTPClientError` from AsyncHTTPClient.
/// So does a cancelled request.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public struct HTTPTransportError: Error, Sendable, CustomStringConvertible {
    /// What went wrong. More kinds can be added, so a `switch` over them needs a `default`.
    public struct Kind: Hashable, Sendable, CustomStringConvertible {
        public let description: String

        private init(_ description: String) {
            self.description = description
        }

        /// The request couldn't be sent as given.
        public static let invalidRequest = Kind("Invalid request")
        /// The answer wasn't an HTTP response.
        public static let invalidResponse = Kind("Invalid response")
        /// The provider answered with an error status: the error has its response and body.
        public static let httpError = Kind("HTTP error")
        /// A successful answer couldn't be read as what was expected.
        public static let decodingError = Kind("Decoding error")
    }

    /// What went wrong.
    public let kind: Kind

    /// The response, headers included, when the provider answered.
    public let response: HTTPResponse?

    /// The response's body, when it was read.
    public let body: Data?

    /// The error that caused this one, such as a decoding error's.
    public let underlyingError: (any Error)?

    public init(
        kind: Kind,
        response: HTTPResponse? = nil,
        body: Data? = nil,
        underlyingError: (any Error)? = nil
    ) {
        self.kind = kind
        self.response = response
        self.body = body
        self.underlyingError = underlyingError
    }

    public var description: String {
        var description = kind.description
        if let response { description += " (Status \(response.status.code))" }
        if let body, let text = String(data: body, encoding: .utf8), !text.isEmpty { description += ": \(text)" }
        if let underlyingError { description += ": \(underlyingError)" }
        return description
    }
}

/// What every transport's requests to a model's provider are made of, written once on top of
/// ``HTTPTransport/send(request:body:timeout:responseHandler:)``: an answer decoded from JSON,
/// one decoded from each line of a stream as it arrives, and one from each event of a stream
/// of server-sent events.
extension HTTPTransport {
    func fetch<T: Decodable>(
        _ request: HTTPRequest,
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) async throws -> T {
        let request = Self.request(request, accepting: "application/json", body: body)
        let data = try await send(request: request, body: body, timeout: nil) { response, body in
            try await Self.validate(response, body)
            return try await body.collect()
        }
        // A body cut short by cancelling isn't read as a whole one, whichever transport sent it.
        try Task.checkCancellation()
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = dateDecodingStrategy
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw HTTPTransportError(kind: .decodingError, underlyingError: error)
        }
    }

    func fetchStream<T: Decodable & Sendable>(
        _ request: HTTPRequest,
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate,
        timeout: Duration? = nil
    ) -> AsyncThrowingStream<T, any Error> {
        let request = Self.request(request, accepting: "application/json", body: body)
        return streaming { continuation in
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = dateDecodingStrategy
            try await send(request: request, body: body, timeout: timeout) { response, body in
                try await Self.validate(response, body)
                // Each line is decoded as it arrives.
                var lines = JSONLines()
                for try await chunk in body {
                    try Task.checkCancellation()
                    for line in lines.append(contentsOf: chunk) {
                        continuation.yield(try Self.decode(T.self, from: line, with: decoder))
                    }
                }
                // What's left is a last line only when the body ended, not when it was cut short.
                try Task.checkCancellation()
                if let line = lines.finish() {
                    continuation.yield(try Self.decode(T.self, from: line, with: decoder))
                }
            }
        }
    }

    func fetchEventStream<T: Decodable & Sendable>(
        _ request: HTTPRequest,
        body: Data? = nil,
        timeout: Duration? = nil
    ) -> AsyncThrowingStream<T, any Error> {
        let request = Self.request(request, accepting: "text/event-stream", body: body)
        return streaming { continuation in
            try await send(request: request, body: body, timeout: timeout) { response, body in
                try await Self.validate(response, body)
                let decoder = JSONDecoder()
                for try await event in body.bytes.events {
                    guard let data = event.data.data(using: .utf8) else { continue }
                    if let decoded = try? decoder.decode(T.self, from: data) {
                        continuation.yield(decoded)
                    }
                }
                // A body that ended because the request was cancelled isn't the end of the stream.
                try Task.checkCancellation()
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
        if body != nil, request.headerFields[.contentType] == nil {
            request.headerFields[.contentType] = "application/json"
        }
        return request
    }

    /// A line of a stream as `T`; a line that isn't one is a decoding error, as a whole answer is.
    private static func decode<T: Decodable>(
        _ type: T.Type,
        from line: some Collection<UInt8>,
        with decoder: JSONDecoder
    ) throws -> T {
        do {
            return try decoder.decode(T.self, from: Data(line))
        } catch {
            throw HTTPTransportError(kind: .decodingError, underlyingError: error)
        }
    }

    /// A response that isn't a success fails with what its body says.
    private static func validate(_ response: HTTPResponse, _ body: HTTPResponseBody) async throws {
        guard response.status.kind == .successful else {
            throw HTTPTransportError(kind: .httpError, response: response, body: try await body.collect())
        }
    }
}
