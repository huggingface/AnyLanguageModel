import Foundation

// Ported from PartialJSONDecoder (https://github.com/mattt/PartialJSONDecoder), Apache-2.0 licensed.

/// An error that occurs while completing partial JSON.
enum JSONCompletionError: Error, Equatable {
    /// The input contains a value that JSON does not allow, such as `NaN` or `Infinity`.
    case invalidValue(String)

    /// The input nests deeper than the completer's limit.
    case depthLimitExceeded(Int)
}

/// Completes partial JSON text by appending the closing characters it is missing.
///
/// The completer scans the text as a JSON value and computes the suffix needed to
/// close every open string, array, and object at the point where the text ends.
/// Unfinished literals such as `tr` or `1.` are completed to `true` and `1.0`,
/// a key with no value receives `null`, and a trailing comma is dropped.
///
/// The result is intended for a JSON parser, not for display:
/// text that is already complete is returned unchanged,
/// and text that is not JSON at all is returned unchanged too.
struct JSONCompleter: Sendable {
    /// The completion for a partial JSON value.
    ///
    /// `string` holds the characters to append,
    /// and `endIndex` is the index in the original text immediately after the portion to keep.
    /// The kept portion can be shorter than the input when the text ends inside an escape sequence.
    typealias Completion = (string: String, endIndex: String.Index)

    /// The maximum number of nested arrays and objects the completer accepts before it throws.
    ///
    /// This bounds recursion on adversarial or malformed input.
    var maximumDepth: Int = 64

    /// Creates a completer with the default depth limit.
    init() {}

    /// Returns the input with any missing closing characters appended.
    ///
    /// - Parameter json: Partial JSON text.
    /// - Returns: Text that closes every structure the input left open.
    /// - Throws: ``JSONCompletionError`` if the input contains a non-JSON literal
    ///   or nests deeper than ``maximumDepth``.
    func complete(_ json: String) throws -> String {
        guard !json.isEmpty else { return "" }

        if let completion = try completion(for: json, from: json.startIndex) {
            return json[..<completion.endIndex] + completion.string
        }

        return json
    }

    /// Returns the completion for the JSON value that starts at the given index.
    ///
    /// - Parameters:
    ///   - json: Partial JSON text.
    ///   - startIndex: The index at which to begin scanning.
    /// - Returns: The completion, or `nil` if the value is already complete
    ///   or no value starts at the index.
    /// - Throws: ``JSONCompletionError`` if the input contains a non-JSON literal
    ///   or nests deeper than ``maximumDepth``.
    func completion(for json: String, from startIndex: String.Index) throws -> Completion? {
        let start = skipWhitespace(json, from: startIndex)
        guard start < json.endIndex else { return nil }

        return try completeValue(json, from: start, depth: 0)
    }

    // MARK: -

    private func skipWhitespace(_ json: String, from index: String.Index) -> String.Index {
        var current = index
        while current < json.endIndex && json[current].isWhitespace {
            current = json.index(after: current)
        }
        return current
    }

    private func completeValue(_ json: String, from startIndex: String.Index, depth: Int) throws -> Completion? {
        guard depth < maximumDepth else {
            throw JSONCompletionError.depthLimitExceeded(maximumDepth)
        }

        let start = skipWhitespace(json, from: startIndex)
        guard start < json.endIndex else { return nil }

        switch json[start] {
        case "{":
            return try completeObject(json, from: start, depth: depth + 1)
        case "[":
            return try completeArray(json, from: start, depth: depth + 1)
        case "\"":
            return completeString(json, from: start)
        case "-":
            let next = json.index(after: start)
            if next < json.endIndex, json[next] == "I" {
                throw JSONCompletionError.invalidValue("-Infinity")
            }
            return completeNumber(json, from: start)
        case "0" ... "9":
            return completeNumber(json, from: start)
        case "t":
            return completeLiteral(json, from: start, literal: "true")
        case "f":
            return completeLiteral(json, from: start, literal: "false")
        case "n":
            return completeLiteral(json, from: start, literal: "null")
        case "I":
            throw JSONCompletionError.invalidValue("Infinity")
        case "N":
            throw JSONCompletionError.invalidValue("NaN")
        default:
            return nil
        }
    }

    /// Completes a string that starts at the given index.
    ///
    /// If the text ends inside an escape sequence, the completion keeps only the
    /// text before the backslash so that the result is valid JSON.
    private func completeString(_ json: String, from startIndex: String.Index) -> Completion? {
        guard startIndex < json.endIndex, json[startIndex] == "\"" else { return nil }

        var current = json.index(after: startIndex)
        while current < json.endIndex {
            let char = json[current]
            if char == "\\" {
                let escapeStart = current
                current = json.index(after: current)
                guard current < json.endIndex else {
                    return (string: "\"", endIndex: escapeStart)
                }
                if json[current] == "u" {
                    var remaining = 4
                    current = json.index(after: current)
                    while remaining > 0 && current < json.endIndex && json[current].isHexDigit {
                        current = json.index(after: current)
                        remaining -= 1
                    }
                    if remaining > 0 {
                        return (string: "\"", endIndex: escapeStart)
                    }
                    continue
                }
            } else if char == "\"" {
                return nil
            }
            current = json.index(after: current)
        }

        return (string: "\"", endIndex: current)
    }

    private func completeArray(_ json: String, from startIndex: String.Index, depth: Int) throws -> Completion? {
        guard startIndex < json.endIndex, json[startIndex] == "[" else { return nil }

        var current = skipWhitespace(json, from: json.index(after: startIndex))
        var requiresComma = false
        var lastValidIndex = current

        if current >= json.endIndex {
            return (string: "]", endIndex: current)
        }

        while current < json.endIndex {
            if json[current] == "]" {
                return nil
            }

            if requiresComma {
                guard json[current] == "," else {
                    return (string: "]", endIndex: lastValidIndex)
                }
                requiresComma = false
                current = skipWhitespace(json, from: json.index(after: current))
                if current >= json.endIndex { break }
                lastValidIndex = current
            }

            if json[current] == "]" {
                return nil
            }

            if let elementCompletion = try completeValue(json, from: current, depth: depth) {
                return (string: elementCompletion.string + "]", endIndex: elementCompletion.endIndex)
            }

            current = findEndOfCompleteValue(json, from: current)
            lastValidIndex = current
            current = skipWhitespace(json, from: current)
            requiresComma = true
        }

        return (string: "]", endIndex: lastValidIndex)
    }

    private func completeObject(_ json: String, from startIndex: String.Index, depth: Int) throws -> Completion? {
        guard startIndex < json.endIndex, json[startIndex] == "{" else { return nil }

        var current = skipWhitespace(json, from: json.index(after: startIndex))
        var requiresComma = false
        var lastValidIndex = current

        if current >= json.endIndex {
            return (string: "}", endIndex: current)
        }

        while current < json.endIndex {
            if json[current] == "}" {
                return nil
            }

            if requiresComma {
                guard json[current] == "," else {
                    return (string: "}", endIndex: lastValidIndex)
                }
                requiresComma = false
                current = skipWhitespace(json, from: json.index(after: current))
                if current >= json.endIndex { break }
                lastValidIndex = current
            }

            if json[current] == "}" {
                return nil
            }

            // Key
            if let keyCompletion = completeString(json, from: current) {
                return (string: keyCompletion.string + ": null}", endIndex: keyCompletion.endIndex)
            }
            let keyEnd = findEndOfCompleteValue(json, from: current)
            guard keyEnd > current else {
                return (string: "}", endIndex: lastValidIndex)
            }
            current = keyEnd
            lastValidIndex = current

            // Colon
            current = skipWhitespace(json, from: current)
            guard current < json.endIndex, json[current] == ":" else {
                return (string: ": null}", endIndex: lastValidIndex)
            }
            current = json.index(after: current)
            lastValidIndex = current

            // Value
            current = skipWhitespace(json, from: current)
            guard current < json.endIndex else {
                return (string: "null}", endIndex: lastValidIndex)
            }

            if let valueCompletion = try completeValue(json, from: current, depth: depth) {
                return (string: valueCompletion.string + "}", endIndex: valueCompletion.endIndex)
            }

            current = findEndOfCompleteValue(json, from: current)
            lastValidIndex = current
            current = skipWhitespace(json, from: current)
            requiresComma = true
        }

        return (string: "}", endIndex: lastValidIndex)
    }

    private func completeNumber(_ json: String, from startIndex: String.Index) -> Completion? {
        var current = startIndex

        if current < json.endIndex && json[current] == "-" {
            current = json.index(after: current)
        }

        guard current < json.endIndex else {
            return (string: "0", endIndex: current)
        }

        if json[current] == "." {
            return (string: "0.0", endIndex: current)
        }

        while current < json.endIndex && json[current].isNumber {
            current = json.index(after: current)
        }

        if current < json.endIndex && json[current] == "." {
            current = json.index(after: current)
            let fractionStart = current
            while current < json.endIndex && json[current].isNumber {
                current = json.index(after: current)
            }
            if current == fractionStart {
                return (string: "0", endIndex: current)
            }
        }

        if current < json.endIndex && (json[current] == "e" || json[current] == "E") {
            current = json.index(after: current)
            if current < json.endIndex && (json[current] == "+" || json[current] == "-") {
                current = json.index(after: current)
            }
            if current >= json.endIndex || !json[current].isNumber {
                return (string: "0", endIndex: current)
            }
            while current < json.endIndex && json[current].isNumber {
                current = json.index(after: current)
            }
        }

        return nil
    }

    private func completeLiteral(_ json: String, from startIndex: String.Index, literal: String) -> Completion? {
        var current = startIndex
        var remaining = literal[...]

        while current < json.endIndex, let expected = remaining.first {
            guard json[current] == expected else { return nil }
            current = json.index(after: current)
            remaining = remaining.dropFirst()
        }

        guard !remaining.isEmpty else { return nil }
        return (string: String(remaining), endIndex: current)
    }

    /// Returns the index immediately after the complete value that starts at the given index.
    ///
    /// Callers invoke this only after `completeValue` has reported the value complete,
    /// so this scans the text once without re-running the completer on it.
    private func findEndOfCompleteValue(_ json: String, from startIndex: String.Index) -> String.Index {
        let start = skipWhitespace(json, from: startIndex)
        guard start < json.endIndex else { return start }

        switch json[start] {
        case "\"":
            var current = json.index(after: start)
            var isEscaped = false
            while current < json.endIndex {
                let char = json[current]
                if char == "\\" {
                    isEscaped.toggle()
                } else if char == "\"" && !isEscaped {
                    return json.index(after: current)
                } else {
                    isEscaped = false
                }
                current = json.index(after: current)
            }
            return current
        case "{":
            return findMatchingBrace(json, from: start, open: "{", close: "}")
        case "[":
            return findMatchingBrace(json, from: start, open: "[", close: "]")
        case "t" where json[start...].hasPrefix("true"):
            return json.index(start, offsetBy: 4)
        case "f" where json[start...].hasPrefix("false"):
            return json.index(start, offsetBy: 5)
        case "n" where json[start...].hasPrefix("null"):
            return json.index(start, offsetBy: 4)
        case "-", "0" ... "9":
            var current = start
            while current < json.endIndex && "0123456789.-+eE".contains(json[current]) {
                current = json.index(after: current)
            }
            return current
        default:
            return start
        }
    }

    /// Returns the index immediately after the bracket that closes the one at the given index.
    private func findMatchingBrace(
        _ json: String,
        from startIndex: String.Index,
        open: Character,
        close: Character
    ) -> String.Index {
        var level = 0
        var current = startIndex
        var inString = false
        var isEscaped = false

        while current < json.endIndex {
            let char = json[current]

            if inString {
                if char == "\\" {
                    isEscaped.toggle()
                } else if char == "\"" && !isEscaped {
                    inString = false
                } else {
                    isEscaped = false
                }
            } else if char == "\"" {
                inString = true
                isEscaped = false
            } else if char == open {
                level += 1
            } else if char == close {
                level -= 1
                if level == 0 {
                    return json.index(after: current)
                }
            }
            current = json.index(after: current)
        }

        return current
    }
}
