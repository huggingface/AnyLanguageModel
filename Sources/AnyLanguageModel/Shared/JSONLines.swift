/// Splits a JSON Lines (NDJSON) body into its lines as the bytes arrive: each line ends with `\n`,
/// and the last one may not. Empty lines are skipped. A `\r` before the `\n` stays in the line,
/// where a JSON decoder reads it as whitespace.
struct JSONLines {
    private var line: [UInt8] = []

    /// The line `byte` completes, if it's the `\n` that ends one.
    mutating func append(_ byte: UInt8) -> [UInt8]? {
        guard byte == UInt8(ascii: "\n") else {
            line.append(byte)
            return nil
        }
        defer { line.removeAll(keepingCapacity: true) }
        return line.isEmpty ? nil : line
    }

    /// The lines `bytes` complete.
    mutating func append(contentsOf bytes: some Sequence<UInt8>) -> [[UInt8]] {
        bytes.compactMap { append($0) }
    }

    /// The last line, once the body has ended, if it had no `\n`.
    mutating func finish() -> [UInt8]? {
        defer { line = [] }
        return line.isEmpty ? nil : line
    }
}
