import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("JSONCompleter")
struct JSONCompleterTests {
    let completer = JSONCompleter()

    @Test func leavesCompleteJSONUnchanged() throws {
        #expect(try completer.complete("42") == "42")
        #expect(try completer.complete("\"hello\"") == "\"hello\"")
        #expect(try completer.complete("[1, 2, 3]") == "[1, 2, 3]")
        #expect(try completer.complete("{\"a\": 1}") == "{\"a\": 1}")
        #expect(try completer.complete("true") == "true")
        #expect(try completer.complete("null") == "null")
    }

    @Test func closesOpenStructures() throws {
        #expect(try completer.complete("[1, 2, 3") == "[1, 2, 3]")
        #expect(try completer.complete("{\"a\": 1") == "{\"a\": 1}")
        #expect(try completer.complete("\"hello") == "\"hello\"")
        #expect(
            try completer.complete(#"{"name": "Alice", "age": 30, "hobbies": ["reading", "hiking"#)
                == #"{"name": "Alice", "age": 30, "hobbies": ["reading", "hiking"]}"#
        )
    }

    @Test func reportsCompletionSuffix() throws {
        #expect(try completer.completion(for: "[1, 2, 3]", from: "[1, 2, 3]".startIndex) == nil)
        #expect(try completer.completion(for: "{\"a\": 1}", from: "{\"a\": 1}".startIndex) == nil)

        #expect(try completer.completion(for: "[1, 2, 3", from: "[1, 2, 3".startIndex)?.string == "]")
        #expect(try completer.completion(for: "{\"a\": 1", from: "{\"a\": 1".startIndex)?.string == "}")
        #expect(try completer.completion(for: "\"hello", from: "\"hello".startIndex)?.string == "\"")

        let nested = "{\"obj\": {\"arr\": [1, 2,"
        #expect(try completer.completion(for: nested, from: nested.startIndex)?.string == "]}}")

        let partial = "{\"complete\": true, \"partial\": {\"arr\": [1, 2,"
        let midIndex = partial.lastIndex(of: "{")!
        #expect(try completer.completion(for: partial, from: midIndex)?.string == "]}")
    }

    @Test func enforcesDepthLimit() throws {
        var limited = JSONCompleter()
        limited.maximumDepth = 10

        #expect(throws: JSONCompletionError.depthLimitExceeded(10)) {
            try limited.complete(String(repeating: "[", count: 20))
        }
        #expect(try limited.complete(String(repeating: "[", count: 5)) == "[[[[[]]]]]")
        #expect(JSONCompleter().maximumDepth >= 32)
    }

    @Test func rejectsNonJSONNumbers() {
        #expect(throws: JSONCompletionError.invalidValue("NaN")) { try completer.complete("[NaN") }
        #expect(throws: JSONCompletionError.invalidValue("Infinity")) { try completer.complete("Infinity") }
        #expect(throws: JSONCompletionError.invalidValue("-Infinity")) { try completer.complete("{\"a\": -Inf") }
    }

    @Test func closesComplexNestedStructure() throws {
        let partial = """
            {
              "name": "Complex Test",
              "data": {
                "numbers": [1, 2, 3, 4, 5],
                "boolean": true,
                "nested": {
                  "array": [
                    {"id": 1, "value": "first"},
                    {"id": 2, "value": "second"},
                    {"id": 3, "value": "third"
            """
        let completed = try completer.complete(partial)
        #expect(completed.hasSuffix("}]}}}"))
        #expect(isValidJSON(completed))
    }

    @Test func handlesEscapes() throws {
        let escaped = #""Special \"quoted\" and \n newline and \t tab and ♥ unicode"#
        #expect(try completer.complete(escaped) == escaped + "\"")

        // Text that ends inside an escape sequence is cut back to the escape start.
        #expect(try completer.complete(#""Partial escape: \"#) == #""Partial escape: ""#)
        #expect(try completer.complete(#""Unicode escape: \u26"#) == #""Unicode escape: ""#)
        #expect(try completer.complete(#"{"a": "esc\"#) == #"{"a": "esc"}"#)
        #expect(try completer.complete(#"{"a": "done\" next"#) == #"{"a": "done\" next"}"#)
        #expect(isValidJSON(try completer.complete(#"{"a": "x\\"#)))
    }

    @Test func handlesEmptyAndWhitespaceInput() throws {
        #expect(try completer.complete("") == "")
        #expect(try completer.complete("   ") == "   ")
        #expect(try completer.complete("\n\t\r ") == "\n\t\r ")
        #expect(try completer.complete("{  ") == "{  }")
        #expect(try completer.complete("[  ") == "[  ]")
    }

    @Test func completesPartialNumbers() throws {
        #expect(try completer.complete("123") == "123")
        #expect(try completer.complete("123.") == "123.0")
        #expect(try completer.complete("123.4") == "123.4")
        #expect(try completer.complete("-") == "-0")
        #expect(try completer.complete("-.") == "-0.0")
        #expect(try completer.complete("-123.") == "-123.0")
        #expect(try completer.complete("1.23e") == "1.23e0")
        #expect(try completer.complete("1.23e+") == "1.23e+0")
        #expect(try completer.complete("1.23e-") == "1.23e-0")
    }

    @Test func completesPartialLiterals() throws {
        #expect(try completer.complete("{\"a\": 1, \"b\": tr") == "{\"a\": 1, \"b\": true}")
        #expect(try completer.complete("[fa") == "[false]")
        #expect(try completer.complete("{\"a\": n") == "{\"a\": null}")
    }

    @Test func completesObjectsWithMissingValues() throws {
        #expect(try completer.complete("{\"key\":") == "{\"key\":null}")
        #expect(try completer.complete("{\"key\": \"value") == "{\"key\": \"value\"}")
        #expect(try completer.complete("{\"key1\": true, \"key2\":") == "{\"key1\": true, \"key2\":null}")
        #expect(try completer.complete("{\"key\": 42,") == "{\"key\": 42}")
        #expect(try completer.complete("{\"ke") == "{\"ke\": null}")
        #expect(
            try completer.complete("{\"outer\": {\"inner\": [1, 2, {\"nested\":")
                == "{\"outer\": {\"inner\": [1, 2, {\"nested\":null}]}}"
        )
    }

    @Test func keepsValuesAfterEmptyNestedContainers() throws {
        #expect(try completer.complete(#"{"a": {}, "b": 2"#) == #"{"a": {}, "b": 2}"#)
        #expect(try completer.complete(#"{"a": [], "b": 2"#) == #"{"a": [], "b": 2}"#)
        #expect(try completer.complete(#"[{}, 2"#) == #"[{}, 2]"#)
        #expect(try completer.complete(#"[[], [1"#) == #"[[], [1]]"#)
        #expect(try completer.complete(#"{"a": {  }, "b""#) == #"{"a": {  }, "b": null}"#)
        #expect(try completer.complete("{}") == "{}")
        #expect(try completer.complete("[]") == "[]")
        #expect(try completer.completion(for: "{}", from: "{}".startIndex) == nil)
        #expect(try completer.completion(for: "[ ]", from: "[ ]".startIndex) == nil)
    }

    @Test func keepsValuesAfterWhitespaceBeforeComma() throws {
        #expect(try completer.complete("[1 , 2") == "[1 , 2]")
        #expect(try completer.complete("[1 ,") == "[1]")
        #expect(try completer.complete("[1 ") == "[1]")
        #expect(try completer.complete(#"{"a": 1 , "b": 2"#) == #"{"a": 1 , "b": 2}"#)
        #expect(
            try completer.complete(#"{"a": "x" , "b": [1 , {"c": true , "d""#)
                == #"{"a": "x" , "b": [1 , {"c": true , "d": null}]}"#
        )
        #expect(
            try completer.complete("{\"a\": 1\n,\n\"b\": [\n1\n,\n2\n")
                == "{\"a\": 1\n,\n\"b\": [\n1\n,\n2]}"
        )
    }

    @Test func completesArraysWithMissingValues() throws {
        #expect(try completer.complete("[1, 2, 3,") == "[1, 2, 3]")
        #expect(try completer.complete("[1, 2,") == "[1, 2]")
        #expect(try completer.complete("[[1, 2], [3,") == "[[1, 2], [3]]")
    }

    private func isValidJSON(_ json: String) -> Bool {
        (try? JSONSerialization.jsonObject(with: Data(json.utf8), options: [.fragmentsAllowed])) != nil
    }
}
