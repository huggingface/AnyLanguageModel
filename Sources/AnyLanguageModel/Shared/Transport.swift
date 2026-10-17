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
/// AnyLanguageModel's remote models send every request through one, given as their `session`,
/// and read its response in ``perform(request:body:timeout:responseHandler:)``:
/// a streamed response is read there as it arrives, to its end.
/// `URLSession` is one, and so is AsyncHTTPClient's `HTTPClient` with the `AsyncHTTPClient` trait.
///
/// Supply your own to decide how requests are sent,
/// such as one that waits for a free slot of an account that takes few requests at once,
/// or one that observes them, then passes each on to another session.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public protocol HTTPTransport: Sendable {
    /// Sends `request` with `body`, and passes its response to `responseHandler`.
    ///
    /// The response's body can be read only in `responseHandler`;
    /// the request is over once it returns.
    ///
    /// - Parameters:
    ///   - request: The request to send.
    ///   - body: What the request sends, if anything.
    ///   - timeout: How long to wait for the response to start.
    ///   - responseHandler: Reads the response, its body as it arrives.
    func perform<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return
}

/// The body of an HTTP response, as its bytes arrive.
///
/// It's read once, in the response handler it's given to.
public struct HTTPResponseBody: AsyncSequence {
    public typealias Element = UInt8

    private let source: ByteSource

    /// A body read from `bytes`.
    public init<Bytes: AsyncSequence>(_ bytes: Bytes) where Bytes.Element == UInt8 {
        source = Iterating(bytes.makeAsyncIterator())
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(source: source)
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        fileprivate let source: ByteSource

        public mutating func next() async throws -> UInt8? {
            try await source.next()
        }
    }

    /// The whole body, as data.
    public func collect() async throws -> Data {
        var data = Data()
        for try await byte in self { data.append(byte) }
        return data
    }
}

/// Where a body's bytes come from, whatever sequence gives them.
///
/// A base class with a generic subclass hides the sequence's type, so a body is one type whatever
/// the session reads it from. `any AsyncIteratorProtocol<UInt8, any Error>` would say the same
/// directly, but it needs macOS 15 and the package supports macOS 14: once it doesn't, this and
/// ``Iterating`` become that existential.
private class ByteSource {
    func next() async throws -> UInt8? { nil }
}

private final class Iterating<Iterator: AsyncIteratorProtocol>: ByteSource where Iterator.Element == UInt8 {
    private var iterator: Iterator

    init(_ iterator: Iterator) {
        self.iterator = iterator
    }

    override func next() async throws -> UInt8? {
        try await iterator.next()
    }
}

#if canImport(AsyncHTTPClient)
    /// The client a remote model sends its requests with unless it's given another transport.
    public typealias HTTPSession = HTTPClient

    public func makeDefaultSession() -> HTTPSession {
        return HTTPClient.shared
    }
#else
    /// The session a remote model sends its requests with unless it's given another transport.
    public typealias HTTPSession = URLSession

    public func makeDefaultSession() -> HTTPSession {
        return URLSession(configuration: .default)
    }
#endif
