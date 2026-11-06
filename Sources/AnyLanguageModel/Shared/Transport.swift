import Foundation
import HTTPTypes

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
#endif

/// How a model sends its HTTP requests.
///
/// AnyLanguageModel's remote models send every request through one, given as their `transport`,
/// and read its response in ``send(request:body:timeout:responseHandler:)``:
/// a streamed response is read there as it arrives, to its end.
/// `URLSession` is one, and so is AsyncHTTPClient's `HTTPClient` with the `AsyncHTTPClient` trait.
///
/// Supply your own to decide how requests are sent,
/// such as one that waits for a free slot of an account that takes few requests at once,
/// or one that observes them, then passes each on to another transport.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public protocol HTTPTransport: Sendable {
    /// Sends `request` with `body`, and passes its response to `responseHandler`.
    ///
    /// A transport keeps to this contract:
    /// - It calls `responseHandler` at most once, with the response to `request`.
    /// - Cancelling the calling task cancels the request, and `send` throws:
    ///   a body cut short by cancelling throws as it's read, rather than ending as if it were whole.
    /// - The response's body can be read only in `responseHandler`:
    ///   once it returns, the request is over, and the body can't be read anymore.
    /// - A transport that retries a request doesn't retry it
    ///   once it has passed a response to `responseHandler`.
    ///
    /// - Parameters:
    ///   - request: The request to send.
    ///   - body: What the request sends, if anything.
    ///   - timeout: How long to wait for the response to start, or `nil` when the caller didn't
    ///     say, for the transport to decide.
    ///     The transport decides when that time starts,
    ///     so a request that waits for a free slot before it's sent doesn't use up its deadline.
    ///     AsyncHTTPClient applies it only until the response starts, and waits 180 seconds without
    ///     one. A URL session applies it to every pause in the response too, and without one waits
    ///     as long as its own `timeoutIntervalForRequest`.
    ///   - responseHandler: Reads the response, its body as it arrives.
    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return
}

/// The body of an HTTP response, in chunks of data as they arrive.
///
/// It's read once, in the response handler it's given to.
public struct HTTPResponseBody: AsyncSequence {
    public typealias Element = Data

    private let source: ChunkSource

    /// A body read from `chunks`.
    public init<Chunks: AsyncSequence>(_ chunks: Chunks) where Chunks.Element == Data {
        source = Iterating(chunks.makeAsyncIterator())
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(source: source)
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        fileprivate let source: ChunkSource

        public mutating func next() async throws -> Data? {
            try await source.next()
        }
    }

    /// The whole body, as data.
    public func collect() async throws -> Data {
        var data = Data()
        for try await chunk in self { data.append(chunk) }
        return data
    }

    /// The body's bytes, one at a time, for a reader that takes bytes:
    /// it waits only for each chunk.
    var bytes: Bytes { Bytes(chunks: self) }

    struct Bytes: AsyncSequence {
        typealias Element = UInt8
        let chunks: HTTPResponseBody

        func makeAsyncIterator() -> Iterator {
            Iterator(chunks: chunks.makeAsyncIterator())
        }

        struct Iterator: AsyncIteratorProtocol {
            var chunks: HTTPResponseBody.AsyncIterator
            var current = Data()
            var index = 0

            mutating func next() async throws -> UInt8? {
                while index == current.count {
                    guard let chunk = try await chunks.next() else { return nil }
                    current = chunk
                    index = 0
                }
                defer { index += 1 }
                return current[current.startIndex + index]
            }
        }
    }
}

/// Where a body's chunks come from, whatever sequence gives them.
///
/// A base class with a generic subclass hides the sequence's type, so a body is one type whatever
/// the transport reads it from. `any AsyncIteratorProtocol<Data, any Error>` would say the same
/// directly, but it needs macOS 15 and the package supports macOS 14: once it doesn't, this and
/// ``Iterating`` become that existential.
private class ChunkSource {
    func next() async throws -> Data? { nil }
}

private final class Iterating<Iterator: AsyncIteratorProtocol>: ChunkSource where Iterator.Element == Data {
    private var iterator: Iterator

    init(_ iterator: Iterator) {
        self.iterator = iterator
    }

    override func next() async throws -> Data? {
        try await iterator.next()
    }
}

#if canImport(AsyncHTTPClient)
    /// The transport a remote model sends its requests with unless it's given another:
    /// AsyncHTTPClient's shared client.
    public func makeDefaultTransport() -> any HTTPTransport {
        HTTPClient.shared
    }

    /// The client a remote model sent its requests with.
    @available(*, deprecated, message: "Use HTTPTransport, and makeDefaultTransport() for the default.")
    public typealias HTTPSession = HTTPClient

    @available(*, deprecated, message: "Use makeDefaultTransport(), which returns any HTTPTransport.")
    public func makeDefaultSession() -> HTTPSession {
        return HTTPClient.shared
    }
#elseif os(WASI)
    /// WASI has no networking of its own, so a remote model needs its `transport` given,
    /// such as one that sends requests with the host's `fetch`.
    public func makeDefaultTransport() -> any HTTPTransport {
        MissingTransport()
    }

    /// The transport a remote model sent its requests with.
    @available(*, deprecated, message: "Use HTTPTransport, and makeDefaultTransport() for the default.")
    public typealias HTTPSession = any HTTPTransport

    private struct MissingTransport: HTTPTransport {
        func send<Return>(
            request: HTTPRequest,
            body: Data?,
            timeout: Duration?,
            responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
        ) async throws -> Return {
            throw HTTPTransportError(kind: .invalidRequest, underlyingError: WASITransportError())
        }
    }

    private struct WASITransportError: Error, CustomStringConvertible {
        var description: String { "WASI has no default transport: give the model a `transport`." }
    }
#else
    /// The transport a remote model sends its requests with unless it's given another:
    /// a URL session with the default configuration.
    public func makeDefaultTransport() -> any HTTPTransport {
        URLSession(configuration: .default)
    }

    /// The session a remote model sent its requests with.
    @available(*, deprecated, message: "Use HTTPTransport, and makeDefaultTransport() for the default.")
    public typealias HTTPSession = URLSession

    @available(*, deprecated, message: "Use makeDefaultTransport(), which returns any HTTPTransport.")
    public func makeDefaultSession() -> HTTPSession {
        return URLSession(configuration: .default)
    }
#endif
