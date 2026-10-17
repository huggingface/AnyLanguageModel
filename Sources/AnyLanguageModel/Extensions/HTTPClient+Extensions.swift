#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
    import Foundation
    import HTTPTypes
    import NIOCore
    import NIOHTTP1

    /// AsyncHTTPClient sends a model's requests, reading each response's body as it arrives.
    extension HTTPClient: HTTPTransport {
        public func perform<Return>(
            request: HTTPRequest,
            body: Data?,
            timeout: Duration,
            responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
        ) async throws -> Return {
            guard let scheme = request.scheme, let authority = request.authority, let path = request.path else {
                throw HTTPTransportError.invalidRequest
            }
            var sent = HTTPClientRequest(url: "\(scheme)://\(authority)\(path)")
            sent.method = HTTPMethod(rawValue: request.method.rawValue)
            for field in request.headerFields {
                sent.headers.add(name: field.name.rawName, value: field.value)
            }
            if let body {
                sent.body = .bytes(ByteBuffer(bytes: body))
            }

            let (seconds, attoseconds) = timeout.components
            let received = try await execute(
                sent,
                timeout: .nanoseconds(seconds * 1_000_000_000 + attoseconds / 1_000_000_000)
            )
            var fields = HTTPFields()
            for (name, value) in received.headers {
                guard let name = HTTPField.Name(name) else { continue }
                fields.append(HTTPField(name: name, value: value))
            }
            let response = HTTPResponse(
                status: HTTPResponse.Status(
                    code: Int(received.status.code),
                    reasonPhrase: received.status.reasonPhrase
                ),
                headerFields: fields
            )
            return try await responseHandler(response, HTTPResponseBody(Bytes(received.body)))
        }
    }

    /// A response body's bytes, one at a time from each buffer as it arrives.
    private struct Bytes<Buffers: AsyncSequence>: AsyncSequence where Buffers.Element == ByteBuffer {
        typealias Element = UInt8
        let buffers: Buffers

        init(_ buffers: Buffers) {
            self.buffers = buffers
        }

        func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(buffers: buffers.makeAsyncIterator())
        }

        struct AsyncIterator: AsyncIteratorProtocol {
            var buffers: Buffers.AsyncIterator
            var current = ByteBuffer()

            mutating func next() async throws -> UInt8? {
                while current.readableBytes == 0 {
                    guard let buffer = try await buffers.next() else { return nil }
                    current = buffer
                }
                return current.readInteger(as: UInt8.self)
            }
        }
    }
#endif
