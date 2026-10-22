import Foundation

#if canImport(Glibc)
    import Glibc
#elseif canImport(Darwin)
    import Darwin
#endif

/// A server on the loopback interface that answers every connection the same way, and then says
/// nothing more: for what a client does while a provider is slow or stops partway.
final class LoopbackServer: @unchecked Sendable {
    enum Answer {
        /// Nothing at all: the response never starts.
        case silence
        /// A response's head, and this much of its body, out of a longer one.
        case partial(contentLength: Int, body: String)
    }

    let port: UInt16
    private let socket: Int32
    private let answer: Answer
    private let lock = NSLock()
    private var connections: [Int32] = []

    init(_ answer: Answer) throws {
        self.answer = answer
        #if canImport(Glibc)
            let fd = Glibc.socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #else
            let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        #endif
        socket = fd
        var address = sockaddr_in()
        #if canImport(Darwin)
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        #endif
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = 0
        var bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, listen(fd, 16) == 0 else { throw POSIXError(.EADDRNOTAVAIL) }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        bound = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        port = UInt16(bigEndian: address.sin_port)
        let thread = Thread { [self] in serve() }
        thread.start()
    }

    var url: URL { URL(string: "http://127.0.0.1:\(port)/")! }

    private func serve() {
        while true {
            let connection = accept(socket, nil, nil)
            guard connection >= 0 else { return }
            lock.withLock { connections.append(connection) }
            // Reads the request, as far as it comes in one read.
            var buffer = [UInt8](repeating: 0, count: 65536)
            _ = read(connection, &buffer, buffer.count)
            if case .partial(let contentLength, let body) = answer {
                let head =
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(contentLength)\r\n\r\n"
                let bytes = Array((head + body).utf8)
                _ = bytes.withUnsafeBytes { write(connection, $0.baseAddress, $0.count) }
            }
        }
    }

    func stop() {
        shutdown(socket, Int32(SHUT_RDWR))
        close(socket)
        for connection in lock.withLock({ connections }) { close(connection) }
    }
}
