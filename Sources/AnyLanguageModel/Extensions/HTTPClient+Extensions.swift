#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
    import Foundation
    import HTTPTypes
    import NIOCore
    import NIOHTTP1

    /// AsyncHTTPClient sends a model's requests, reading each response's body as it arrives.
    extension HTTPClient: HTTPTransport {
        public func send<Return>(
            request: HTTPRequest,
            body: Data?,
            timeout: Duration,
            responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
        ) async throws -> Return {
            guard let scheme = request.scheme, let authority = request.authority, let path = request.path else {
                throw HTTPTransportError(kind: .invalidRequest)
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
            return try await responseHandler(
                response,
                HTTPResponseBody(received.body.map { Data($0.readableBytesView) })
            )
        }
    }
#endif
