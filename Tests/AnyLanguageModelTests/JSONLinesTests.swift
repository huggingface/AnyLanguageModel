import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("JSON Lines")
struct JSONLinesTests {
    private func split(_ chunks: [String]) -> [String] {
        var lines = JSONLines()
        var split = chunks.flatMap { lines.append(contentsOf: Array($0.utf8)) }
        if let last = lines.finish() { split.append(last) }
        return split.map { String(decoding: $0, as: UTF8.self) }
    }

    @Test func linesSplitAcrossChunksAreJoined() {
        #expect(split([#"{"a":"#, #"1}"#, "\n{\"b\":2}\n"]) == [#"{"a":1}"#, #"{"b":2}"#])
    }

    @Test func aLastLineWithoutNewlineIsKept() {
        #expect(split(["{\"a\":1}\n{\"b\":2}"]) == [#"{"a":1}"#, #"{"b":2}"#])
    }

    @Test func emptyLinesAreSkipped() {
        #expect(split(["\n{\"a\":1}\n\n\n"]) == [#"{"a":1}"#])
    }

    /// The `\r` of a `\r\n` stays, and decodes as whitespace.
    @Test func crlfLinesDecode() throws {
        let lines = split(["{\"a\":1}\r\n"])
        #expect(lines == ["{\"a\":1}\r"])
        struct Line: Decodable { let a: Int }
        #expect(try JSONDecoder().decode(Line.self, from: Data(lines[0].utf8)).a == 1)
    }
}
