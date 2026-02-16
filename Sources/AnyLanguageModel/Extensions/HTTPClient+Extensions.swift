import AsyncHTTPClient
import EventSource
import Foundation
import JSONSchema
import NIOCore
import NIOHTTP1

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

enum HTTP {
    enum Method: String {
        case get = "GET"
        case post = "POST"
    }
}

extension HTTPClient {
    func fetch<T: Decodable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) async throws -> T {
        var request = HTTPClientRequest(url: url.absoluteString)
        request.method = method == .get ? .GET : .POST
        request.headers.add(name: "Accept", value: "application/json")

        for (key, value) in headers {
            request.headers.add(name: key, value: value)
        }

        if let body {
            request.body = .bytes(ByteBuffer(bytes: body))
            request.headers.add(name: "Content-Type", value: "application/json")
        }

        let response = try await self.execute(request, timeout: .seconds(60))

        let statusCode = Int(response.status.code)
        guard (200..<300).contains(statusCode) else {
            let bodyData = try await response.body.collect(upTo: 1024 * 1024) // 1MB limit
            let errorString = String(buffer: bodyData)
            throw HTTPClientError.httpError(statusCode: statusCode, detail: errorString)
        }

        let bodyData = try await response.body.collect(upTo: 10 * 1024 * 1024) // 10MB limit
        var data = Data()
        data.reserveCapacity(bodyData.readableBytes)
        data.append(contentsOf: bodyData.readableBytesView)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = dateDecodingStrategy

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw HTTPClientError.decodingError(detail: error.localizedDescription)
        }
    }

    func fetchStream<T: Decodable & Sendable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil,
        dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = _Concurrency.Task { @Sendable in
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = dateDecodingStrategy

                do {
                    var request = HTTPClientRequest(url: url.absoluteString)
                    request.method = method == .get ? .GET : .POST
                    request.headers.add(name: "Accept", value: "application/json")

                    for (key, value) in headers {
                        request.headers.add(name: key, value: value)
                    }

                    if let body {
                        request.body = .bytes(ByteBuffer(bytes: body))
                        request.headers.add(name: "Content-Type", value: "application/json")
                    }

                    let response = try await self.execute(request, timeout: .seconds(300))

                    let statusCode = Int(response.status.code)
                    guard (200..<300).contains(statusCode) else {
                        let bodyData = try await response.body.collect(upTo: 1024 * 1024)
                        let errorString = String(buffer: bodyData)
                        throw HTTPClientError.httpError(statusCode: statusCode, detail: errorString)
                    }

                    // Collect the full body
                    let bodyData = try await response.body.collect(upTo: 10 * 1024 * 1024)
                    var buffer = Data()
                    buffer.reserveCapacity(bodyData.readableBytes)
                    buffer.append(contentsOf: bodyData.readableBytesView)

                    // Process line by line
                    while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                        let chunk = buffer[..<newlineIndex]
                        buffer = buffer[buffer.index(after: newlineIndex)...]

                        if !chunk.isEmpty {
                            let decoded = try decoder.decode(T.self, from: chunk)
                            continuation.yield(decoded)
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    func fetchEventStream<T: Decodable & Sendable>(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil
    ) -> AsyncThrowingStream<T, any Error> {
        AsyncThrowingStream { continuation in
            let task = _Concurrency.Task { @Sendable in
                do {
                    var request = HTTPClientRequest(url: url.absoluteString)
                    request.method = method == .get ? .GET : .POST
                    request.headers.add(name: "Accept", value: "text/event-stream")

                    for (key, value) in headers {
                        request.headers.add(name: key, value: value)
                    }

                    if let body {
                        request.body = .bytes(ByteBuffer(bytes: body))
                        request.headers.add(name: "Content-Type", value: "application/json")
                    }

                    let response = try await self.execute(request, timeout: .seconds(300))

                    let statusCode = Int(response.status.code)
                    guard (200..<300).contains(statusCode) else {
                        let bodyData = try await response.body.collect(upTo: 1024 * 1024)
                        let errorString = String(buffer: bodyData)
                        throw HTTPClientError.httpError(statusCode: statusCode, detail: errorString)
                    }

                    let decoder = JSONDecoder()

                    // Convert response body to async byte stream and process server-sent events
                    let byteStream = ByteStreamFromHTTPBody(response.body)
                    for try await event in byteStream.events {
                        guard let data = event.data.data(using: .utf8) else { continue }
                        if let decoded = try? decoder.decode(T.self, from: data) {
                            continuation.yield(decoded)
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

// Helper to convert HTTPClientResponse.Body to an AsyncSequence of bytes
private struct ByteStreamFromHTTPBody: AsyncSequence {
    typealias Element = UInt8

    private let body: HTTPClientResponse.Body

    init(_ body: HTTPClientResponse.Body) {
        self.body = body
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(body: body)
    }

    struct AsyncIterator: AsyncIteratorProtocol {
        private var bodyIterator: HTTPClientResponse.Body.AsyncIterator
        private var currentBuffer: ByteBuffer?
        private var currentIndex: Int = 0

        init(body: HTTPClientResponse.Body) {
            self.bodyIterator = body.makeAsyncIterator()
        }

        mutating func next() async throws -> UInt8? {
            // If we have a current buffer with remaining bytes, return the next byte
            if let buffer = currentBuffer, currentIndex < buffer.readableBytes {
                let byte = buffer.getInteger(at: buffer.readerIndex + currentIndex, as: UInt8.self)
                currentIndex += 1
                return byte
            }

            // Otherwise, get the next buffer from the body
            guard let nextBuffer = try await bodyIterator.next() else {
                return nil
            }

            currentBuffer = nextBuffer
            currentIndex = 0

            // Return the first byte from the new buffer
            if nextBuffer.readableBytes > 0 {
                let byte = nextBuffer.getInteger(at: nextBuffer.readerIndex, as: UInt8.self)
                currentIndex += 1
                return byte
            }

            // If the buffer is empty, try again
            return try await next()
        }
    }
}

enum HTTPClientError: Error, CustomStringConvertible {
    case invalidResponse
    case httpError(statusCode: Int, detail: String)
    case decodingError(detail: String)

    var description: String {
        switch self {
        case .invalidResponse:
            return "Invalid response"
        case .httpError(let statusCode, let detail):
            return "HTTP error (Status \(statusCode)): \(detail)"
        case .decodingError(let detail):
            return "Decoding error: \(detail)"
        }
    }
}