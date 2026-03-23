#if canImport(AsyncHTTPClient)
    // AsyncHTTPClient.HTTPHandler introduces a Task type that clashes with Swift's Task.
    // Bind Swift's structured-concurrency Task before importing AsyncHTTPClient.
    typealias SwiftTask = Task

    /// Holds the body-read task for `fetchEventStream` so outer stream cancellation can stop NIO body iteration.
    private final class HTTPClientBodyReaderTaskBox: @unchecked Sendable {
        var task: SwiftTask<Void, Never>?
    }

    import AsyncHTTPClient
    import EventSource
    import Foundation
    import NIOCore
    import NIOHTTP1
    import NIOFoundationCompat

    extension HTTPClient {
        func fetch<T: Decodable>(
            _ method: HTTP.Method,
            url: URL,
            headers: [String: String] = [:],
            body: Data? = nil,
            dateDecodingStrategy: JSONDecoder.DateDecodingStrategy = .deferredToDate
        ) async throws -> T {
            var request = HTTPClientRequest(url: url.absoluteString)
            request.method = HTTPMethod(rawValue: method.rawValue)
            request.headers.add(name: "Accept", value: "application/json")

            for (key, value) in headers {
                request.headers.add(name: key, value: value)
            }

            if let body {
                request.body = .bytes(ByteBuffer(data: body))
                request.headers.add(name: "Content-Type", value: "application/json")
            }

            let response = try await self.execute(request, timeout: .seconds(180))

            guard (200 ..< 300).contains(response.status.code) else {
                let bodyData = try await Data(buffer: response.body.collect(upTo: 1024 * 1024))
                if let errorString = String(data: bodyData, encoding: .utf8) {
                    throw HTTPClientError.httpError(statusCode: Int(response.status.code), detail: errorString)
                }
                throw HTTPClientError.httpError(statusCode: Int(response.status.code), detail: "Invalid response")
            }

            let bodyData = try await Data(buffer: response.body.collect(upTo: 1024 * 1024))

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = dateDecodingStrategy

            do {
                return try decoder.decode(T.self, from: bodyData)
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
                let task = SwiftTask { @Sendable in
                    let decoder = JSONDecoder()
                    decoder.dateDecodingStrategy = dateDecodingStrategy

                    do {
                        var request = HTTPClientRequest(url: url.absoluteString)
                        request.method = HTTPMethod(rawValue: method.rawValue)
                        request.headers.add(name: "Accept", value: "application/json")

                        for (key, value) in headers {
                            request.headers.add(name: key, value: value)
                        }

                        if let body {
                            request.body = .bytes(ByteBuffer(data: body))
                            request.headers.add(name: "Content-Type", value: "application/json")
                        }

                        let response = try await self.execute(request, timeout: .seconds(60))

                        guard (200 ..< 300).contains(response.status.code) else {
                            let bodyData = try await Data(buffer: response.body.collect(upTo: 1024 * 1024))
                            if let errorString = String(data: bodyData, encoding: .utf8) {
                                throw HTTPClientError.httpError(
                                    statusCode: Int(response.status.code),
                                    detail: errorString
                                )
                            }
                            throw HTTPClientError.httpError(
                                statusCode: Int(response.status.code),
                                detail: "Invalid response"
                            )
                        }

                        var buffer = Data()

                        for try await chunk in response.body {
                            try SwiftTask.checkCancellation()
                            buffer.append(contentsOf: chunk.readableBytesView)

                            while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                                let line = buffer[..<newlineIndex]
                                buffer = buffer[buffer.index(after: newlineIndex)...]

                                if !line.isEmpty {
                                    let decoded = try decoder.decode(T.self, from: line)
                                    continuation.yield(decoded)
                                }
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
                let bodyReaderBox = HTTPClientBodyReaderTaskBox()

                let task = SwiftTask { @Sendable in
                    do {
                        var request = HTTPClientRequest(url: url.absoluteString)
                        request.method = HTTPMethod(rawValue: method.rawValue)
                        request.headers.add(name: "Accept", value: "text/event-stream")

                        for (key, value) in headers {
                            request.headers.add(name: key, value: value)
                        }

                        if let body {
                            request.body = .bytes(ByteBuffer(data: body))
                            request.headers.add(name: "Content-Type", value: "application/json")
                        }

                        let response = try await self.execute(request, timeout: .seconds(60))

                        guard (200 ..< 300).contains(response.status.code) else {
                            let bodyData = try await Data(buffer: response.body.collect(upTo: 1024 * 1024))
                            if let errorString = String(data: bodyData, encoding: .utf8) {
                                throw HTTPClientError.httpError(
                                    statusCode: Int(response.status.code),
                                    detail: errorString
                                )
                            }
                            throw HTTPClientError.httpError(
                                statusCode: Int(response.status.code),
                                detail: "Invalid response"
                            )
                        }

                        let asyncBytes = AsyncStream<UInt8> { byteContinuation in
                            bodyReaderBox.task = SwiftTask {
                                do {
                                    for try await buffer in response.body {
                                        try SwiftTask.checkCancellation()
                                        for byte in buffer.readableBytesView {
                                            byteContinuation.yield(byte)
                                        }
                                    }
                                    byteContinuation.finish()
                                } catch {
                                    byteContinuation.finish()
                                }
                            }
                            byteContinuation.onTermination = { _ in
                                bodyReaderBox.task?.cancel()
                            }
                        }

                        try await self.decodeAndYieldEventStream(asyncBytes, to: continuation)
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }

                continuation.onTermination = { _ in
                    bodyReaderBox.task?.cancel()
                    task.cancel()
                }
            }
        }

        private func decodeAndYieldEventStream<T: Decodable & Sendable, Bytes>(
            _ asyncBytes: Bytes,
            to continuation: AsyncThrowingStream<T, any Error>.Continuation
        ) async throws where Bytes: AsyncSequence, Bytes.Element == UInt8 {
            let decoder = JSONDecoder()
            for try await event in asyncBytes.events {
                guard let data = event.data.data(using: .utf8) else { continue }
                if let decoded = try? decoder.decode(T.self, from: data) {
                    continuation.yield(decoded)
                }
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
#endif
