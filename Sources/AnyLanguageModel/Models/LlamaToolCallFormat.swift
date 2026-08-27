import Foundation

/// The tool-calling syntax a GGUF model was trained on, detected from its
/// embedded chat template.
///
/// `llama_chat_apply_template` renders chat messages but has no parameter for
/// tool definitions, so tool support is implemented at this layer: definitions
/// are rendered into the system prompt, past tool turns are replayed in the
/// model's native markup, and calls are parsed back out of generated text.
enum LlamaToolCallFormat: Sendable, Equatable {
    /// Hermes-style JSON calls, used by Qwen 2.5/3 and many community fine-tunes:
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
    case hermesJSON

    /// Qwen 3.5 XML calls:
    /// `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`.
    case qwenXML

    /// Gemma 4 canonical calls:
    /// `<|tool_call>call:name{key:value}<tool_call|>`, with `<|"|>`-quoted strings.
    case gemma

    /// Detects the format from a model's embedded chat template text.
    /// Unrecognized templates fall back to the Hermes JSON convention.
    static func detect(template: String?) -> LlamaToolCallFormat {
        guard let template else { return .hermesJSON }
        if template.contains("<|turn>") { return .gemma }
        if template.contains("<function=") { return .qwenXML }
        return .hermesJSON
    }

    /// The marker that ends a complete tool-call block in generated text.
    ///
    /// Generation stops at the first terminator, so each round of a tool
    /// exchange carries exactly one call. Models that want several calls
    /// issue them across consecutive rounds, each with its own output.
    var callTerminator: String {
        switch self {
        case .hermesJSON, .qwenXML: return "</tool_call>"
        case .gemma: return "<tool_call|>"
        }
    }

    /// The marker that starts a tool-call block in generated text.
    var callStartMarker: String {
        switch self {
        case .hermesJSON, .qwenXML: return "<tool_call>"
        case .gemma: return "<|tool_call>"
        }
    }

    /// Markers that open a Gemma 4 thought channel. The canonical template
    /// writes `<|channel>`, but deployed quantizations have been observed
    /// emitting `<|channel|>`, so both spellings are recognized.
    static let gemmaChannelOpenMarkers = ["<|channel>", "<|channel|>"]

    /// The marker that closes a Gemma 4 thought channel.
    static let gemmaChannelCloseMarker = "<channel|>"

    /// Removes Gemma 4 thought-channel spans from generated text. Thinking is
    /// opt-in via `<|think|>`, but the model volunteers thought channels
    /// anyway; the canonical template ships a `strip_thinking` macro for the
    /// same reason. A span left unclosed at the end of the text is removed
    /// through the end.
    static func stripGemmaThoughtChannels(from text: String) -> String {
        var result = ""
        var remainder = Substring(text)
        while let open = earliestRange(of: gemmaChannelOpenMarkers, in: remainder) {
            result += remainder[..<open.lowerBound]
            let afterOpen = remainder[open.upperBound...]
            if let close = afterOpen.range(of: gemmaChannelCloseMarker) {
                remainder = afterOpen[close.upperBound...]
            } else {
                remainder = Substring("")
            }
        }
        result += remainder
        return result
    }

    private static func earliestRange(
        of markers: [String],
        in text: Substring
    ) -> Range<Substring.Index>? {
        var earliest: Range<Substring.Index>?
        for marker in markers {
            if let range = text.range(of: marker),
                earliest == nil || range.lowerBound < earliest!.lowerBound
            {
                earliest = range
            }
        }
        return earliest
    }

    /// The portion of partially generated text that is safe to show while
    /// streaming: completed thought channels are removed (Gemma only), text
    /// from a tool-call start onward is withheld when tools are active, and a
    /// trailing partial match of either marker is held back until the next
    /// token confirms or breaks it.
    func streamingVisibleText(in raw: String, withholdToolCalls: Bool) -> String {
        var text = raw
        if self == .gemma {
            text = Self.stripGemmaThoughtChannels(from: text)
        }
        if withholdToolCalls, let range = text.range(of: callStartMarker) {
            text = String(text[..<range.lowerBound])
        }
        var candidates: [String] = []
        if withholdToolCalls {
            candidates.append(callStartMarker)
        }
        if self == .gemma {
            candidates.append(contentsOf: Self.gemmaChannelOpenMarkers)
        }
        var cut = 0
        for marker in candidates {
            let maxLength = min(marker.count - 1, text.count)
            guard maxLength > 0 else { continue }
            for length in stride(from: maxLength, through: 1, by: -1)
            where text.hasSuffix(String(marker.prefix(length))) {
                cut = max(cut, length)
                break
            }
        }
        if cut > 0 {
            text.removeLast(cut)
        }
        return text
    }
}

/// A tool definition rendered into the system prompt.
struct LlamaToolDefinition {
    let name: String
    let description: String
    let parameters: [String: Any]?
}

/// A tool call parsed out of generated text.
struct LlamaParsedToolCall: Equatable {
    let name: String
    let argumentsJSON: String
}

// MARK: - System prompt rendering

extension LlamaToolCallFormat {
    /// Renders the tool section of the system prompt and merges it with any
    /// existing system text, following each template's own ordering.
    func systemMessage(existingText: String, tools: [LlamaToolDefinition]) -> String {
        guard !tools.isEmpty else { return existingText }
        switch self {
        case .hermesJSON:
            let block = hermesToolsBlock(tools: tools)
            return existingText.isEmpty ? block : existingText + "\n\n" + block
        case .qwenXML:
            let block = qwenXMLToolsBlock(tools: tools)
            return existingText.isEmpty ? block : block + "\n\n" + existingText
        case .gemma:
            let declarations = tools.map { "<|tool>" + gemmaDeclaration(for: $0) + "<tool|>" }.joined()
            return existingText + declarations
        }
    }

    private func toolSpecJSON(_ tool: LlamaToolDefinition) -> String {
        var function: [String: Any] = [
            "name": tool.name,
            "description": tool.description,
        ]
        if let parameters = tool.parameters {
            function["parameters"] = parameters
        }
        let spec: [String: Any] = ["type": "function", "function": function]
        guard
            let data = try? JSONSerialization.data(withJSONObject: spec, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return json
    }

    private func hermesToolsBlock(tools: [LlamaToolDefinition]) -> String {
        var block = "# Tools\n\n"
        block += "You may call one or more functions to assist with the user query.\n\n"
        block += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for tool in tools {
            block += "\n" + toolSpecJSON(tool)
        }
        block += "\n</tools>\n\n"
        block +=
            "For each function call, return a json object with function name and arguments within "
            + "<tool_call></tool_call> XML tags:\n<tool_call>\n"
            + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        return block
    }

    private func qwenXMLToolsBlock(tools: [LlamaToolDefinition]) -> String {
        var block = "# Tools\n\n"
        block += "You have access to the following functions:\n\n<tools>"
        for tool in tools {
            block += "\n" + toolSpecJSON(tool)
        }
        block += "\n</tools>\n\n"
        block += "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        block += "<tool_call>\n<function=example_function_name>\n"
        block += "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        block += "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        block += "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        block += "<IMPORTANT>\nReminder:\n"
        block +=
            "- Function calls MUST follow the specified format: an inner <function=...></function> "
            + "block must be nested within <tool_call></tool_call> XML tags\n"
        block += "- Required parameters MUST be specified\n"
        block +=
            "- You may provide optional reasoning for your function call in natural language "
            + "BEFORE the function call, but NOT after\n"
        block +=
            "- If there is no function call available, answer the question like normal with your "
            + "current knowledge and do not tell the user about function calls\n"
        block += "</IMPORTANT>"
        return block
    }
}

// MARK: - Gemma declaration and argument notation

extension LlamaToolCallFormat {
    /// Renders one Gemma 4 function declaration:
    /// `declaration:name{description:<|"|>...<|"|>,parameters:{...}}`.
    /// Types are uppercased and strings are quoted with the `<|"|>` token, per
    /// the canonical template's `format_function_declaration` macro.
    fileprivate func gemmaDeclaration(for tool: LlamaToolDefinition) -> String {
        var rendered = "declaration:\(tool.name){description:\(gemmaQuote(tool.description))"
        if let parameters = tool.parameters {
            rendered += ",parameters:{"
            var parts: [String] = []
            if let properties = parameters["properties"] as? [String: Any], !properties.isEmpty {
                parts.append("properties:{" + gemmaProperties(properties) + "}")
            }
            if let required = parameters["required"] as? [Any], !required.isEmpty {
                let items = required.map { gemmaQuote("\($0)") }.joined(separator: ",")
                parts.append("required:[\(items)]")
            }
            if let type = parameters["type"] as? String {
                parts.append("type:\(gemmaQuote(type.uppercased()))")
            }
            rendered += parts.joined(separator: ",") + "}"
        }
        rendered += "}"
        return rendered
    }

    private func gemmaProperties(_ properties: [String: Any]) -> String {
        var parts: [String] = []
        for key in properties.keys.sorted() {
            guard let value = properties[key] as? [String: Any] else { continue }
            var fields: [String] = []
            if let description = value["description"] as? String {
                fields.append("description:\(gemmaQuote(description))")
            }
            let type = (value["type"] as? String)?.uppercased() ?? "STRING"
            if type == "STRING", let enumValues = value["enum"] as? [Any] {
                let items = enumValues.map { gemmaArgument($0) }.joined(separator: ",")
                fields.append("enum:[\(items)]")
            }
            if type == "ARRAY", let items = value["items"] as? [String: Any], !items.isEmpty {
                var itemFields: [String] = []
                for itemKey in items.keys.sorted() {
                    guard let itemValue = items[itemKey] else { continue }
                    if itemKey == "type", let itemType = itemValue as? String {
                        itemFields.append("type:\(gemmaQuote(itemType.uppercased()))")
                    } else if itemKey == "properties", let nested = itemValue as? [String: Any] {
                        itemFields.append("properties:{" + gemmaProperties(nested) + "}")
                    } else if itemKey == "required", let required = itemValue as? [Any] {
                        let names = required.map { gemmaQuote("\($0)") }.joined(separator: ",")
                        itemFields.append("required:[\(names)]")
                    } else {
                        itemFields.append("\(itemKey):\(gemmaArgument(itemValue))")
                    }
                }
                fields.append("items:{" + itemFields.joined(separator: ",") + "}")
            }
            if type == "OBJECT", let nested = value["properties"] as? [String: Any] {
                fields.append("properties:{" + gemmaProperties(nested) + "}")
                if let required = value["required"] as? [Any], !required.isEmpty {
                    let names = required.map { gemmaQuote("\($0)") }.joined(separator: ",")
                    fields.append("required:[\(names)]")
                }
            }
            fields.append("type:\(gemmaQuote(type))")
            parts.append("\(key):{" + fields.joined(separator: ",") + "}")
        }
        return parts.joined(separator: ",")
    }

    fileprivate func gemmaQuote(_ string: String) -> String {
        "<|\"|>\(string)<|\"|>"
    }

    /// Renders one JSON value in Gemma 4 argument notation: unquoted keys,
    /// `<|"|>`-quoted strings, and dictionary keys in sorted order.
    fileprivate func gemmaArgument(_ value: Any) -> String {
        switch value {
        case is NSNull:
            return "null"
        case let string as String:
            return gemmaQuote(string)
        case let number as NSNumber:
            if isBooleanNumber(number) {
                return number.boolValue ? "true" : "false"
            }
            if number.doubleValue == number.doubleValue.rounded(),
                number.doubleValue.magnitude < 1e15,
                !"\(number)".contains(".")
            {
                return "\(number.int64Value)"
            }
            return "\(number)"
        case let dictionary as [String: Any]:
            let fields = dictionary.keys.sorted().map { "\($0):\(gemmaArgument(dictionary[$0]!))" }
            return "{" + fields.joined(separator: ",") + "}"
        case let array as [Any]:
            return "[" + array.map { gemmaArgument($0) }.joined(separator: ",") + "]"
        default:
            return gemmaQuote("\(value)")
        }
    }

    /// Renders a JSON object string as Gemma 4 call arguments (the text between
    /// the braces of `call:name{...}`).
    fileprivate func gemmaArgumentsBody(fromJSON json: String) -> String {
        guard
            let data = json.data(using: .utf8),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            return ""
        }
        return object.keys.sorted().map { "\($0):\(gemmaArgument(object[$0]!))" }.joined(separator: ",")
    }
}

// MARK: - Transcript replay rendering

extension LlamaToolCallFormat {
    /// Renders past tool calls as the assistant-message text the model
    /// originally produced, so multi-turn history replays faithfully.
    func assistantText(for calls: [LlamaParsedToolCall], precededByContent: Bool) -> String {
        var parts: [String] = []
        for call in calls {
            switch self {
            case .hermesJSON:
                parts.append(
                    "<tool_call>\n{\"name\": \"\(call.name)\", \"arguments\": \(call.argumentsJSON)}\n</tool_call>"
                )
            case .qwenXML:
                var block = "<tool_call>\n<function=\(call.name)>\n"
                if let data = call.argumentsJSON.data(using: .utf8),
                    let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                {
                    for key in object.keys.sorted() {
                        block += "<parameter=\(key)>\n\(qwenXMLParameterValue(object[key]!))\n</parameter>\n"
                    }
                }
                block += "</function>\n</tool_call>"
                parts.append(block)
            case .gemma:
                parts.append(
                    "<|tool_call>call:\(call.name){\(gemmaArgumentsBody(fromJSON: call.argumentsJSON))}<tool_call|>"
                )
            }
        }
        let joined = parts.joined(separator: "\n")
        if precededByContent && self != .gemma {
            return "\n" + joined
        }
        return joined
    }

    private func qwenXMLParameterValue(_ value: Any) -> String {
        if let string = value as? String { return string }
        if let number = value as? NSNumber {
            if isBooleanNumber(number) {
                return number.boolValue ? "true" : "false"
            }
            return "\(number)"
        }
        guard
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            return "\(value)"
        }
        return json
    }

    /// Renders one tool output as the message that carries it back to the model.
    /// Hermes and Qwen XML formats deliver results inside a user turn; Gemma 4
    /// continues the open model turn with a `<|tool_response>` block.
    func toolResponseMessage(toolName: String, content: String) -> (role: String, content: String) {
        switch self {
        case .hermesJSON, .qwenXML:
            return ("user", "<tool_response>\n\(content)\n</tool_response>")
        case .gemma:
            let body: String
            if let data = content.data(using: .utf8),
                let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            {
                body = object.keys.sorted().map { "\($0):\(gemmaArgument(object[$0]!))" }.joined(separator: ",")
            } else {
                body = "value:\(gemmaArgument(content))"
            }
            return ("tool", "<|tool_response>response:\(toolName){\(body)}<tool_response|>")
        }
    }
}

// MARK: - Parsing generated text

extension LlamaToolCallFormat {
    /// Splits generated text into the visible response and any tool calls,
    /// removing the call markup from the visible portion.
    func parseToolCalls(in text: String) -> (visibleText: String, calls: [LlamaParsedToolCall]) {
        switch self {
        case .hermesJSON:
            return parseMarkedBlocks(in: text, start: "<tool_call>", end: "</tool_call>") { body in
                parseHermesCall(body)
            }
        case .qwenXML:
            return parseMarkedBlocks(in: text, start: "<tool_call>", end: "</tool_call>") { body in
                parseQwenXMLCall(body)
            }
        case .gemma:
            return parseGemmaCalls(in: text)
        }
    }

    private func parseMarkedBlocks(
        in text: String,
        start: String,
        end: String,
        parse: (String) -> LlamaParsedToolCall?
    ) -> (String, [LlamaParsedToolCall]) {
        var visible = ""
        var calls: [LlamaParsedToolCall] = []
        var remainder = Substring(text)
        while let startRange = remainder.range(of: start) {
            visible += remainder[..<startRange.lowerBound]
            let afterStart = remainder[startRange.upperBound...]
            guard let endRange = afterStart.range(of: end) else {
                visible += remainder[startRange.lowerBound...]
                remainder = Substring("")
                break
            }
            let body = String(afterStart[..<endRange.lowerBound])
            if let call = parse(body) {
                calls.append(call)
            }
            remainder = afterStart[endRange.upperBound...]
        }
        visible += remainder
        return (visible.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private func parseHermesCall(_ body: String) -> LlamaParsedToolCall? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let data = trimmed.data(using: .utf8),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let name = object["name"] as? String
        else {
            return nil
        }
        var argumentsJSON = "{}"
        if let arguments = object["arguments"] {
            if let nested = arguments as? String {
                argumentsJSON = nested
            } else if let argumentsData = try? JSONSerialization.data(
                withJSONObject: arguments,
                options: [.sortedKeys]
            ), let json = String(data: argumentsData, encoding: .utf8) {
                argumentsJSON = json
            }
        }
        return LlamaParsedToolCall(name: name, argumentsJSON: argumentsJSON)
    }

    private func parseQwenXMLCall(_ body: String) -> LlamaParsedToolCall? {
        guard let nameStart = body.range(of: "<function=") else { return nil }
        let afterName = body[nameStart.upperBound...]
        guard let nameEnd = afterName.firstIndex(of: ">") else { return nil }
        let name = String(afterName[..<nameEnd])
        guard !name.isEmpty else { return nil }

        var arguments: [String: Any] = [:]
        var remainder = afterName[afterName.index(after: nameEnd)...]
        while let paramStart = remainder.range(of: "<parameter=") {
            let afterParam = remainder[paramStart.upperBound...]
            guard let keyEnd = afterParam.firstIndex(of: ">") else { break }
            let key = String(afterParam[..<keyEnd])
            let valueStart = afterParam.index(after: keyEnd)
            guard let paramEnd = afterParam[valueStart...].range(of: "</parameter>") else { break }
            var value = String(afterParam[valueStart ..< paramEnd.lowerBound])
            if value.hasPrefix("\n") { value.removeFirst() }
            if value.hasSuffix("\n") { value.removeLast() }
            arguments[key] = qwenXMLDecodedValue(value)
            remainder = afterParam[paramEnd.upperBound...]
        }

        guard
            let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        return LlamaParsedToolCall(name: name, argumentsJSON: json)
    }

    /// The XML format writes objects and arrays as JSON but scalars as raw
    /// text, so structured values are decoded and everything else stays a
    /// string.
    private func qwenXMLDecodedValue(_ raw: String) -> Any {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("{") || trimmed.hasPrefix("[") else { return raw }
        guard
            let data = trimmed.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else {
            return raw
        }
        return object
    }

    private func parseGemmaCalls(in text: String) -> (String, [LlamaParsedToolCall]) {
        var visible = ""
        var calls: [LlamaParsedToolCall] = []
        var remainder = Substring(text)
        while let startRange = remainder.range(of: "<|tool_call>call:") {
            visible += remainder[..<startRange.lowerBound]
            let afterStart = remainder[startRange.upperBound...]
            guard let braceIndex = afterStart.firstIndex(of: "{") else {
                visible += remainder[startRange.lowerBound...]
                remainder = Substring("")
                break
            }
            let name = String(afterStart[..<braceIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            let bodyStart = afterStart.index(after: braceIndex)
            guard let bodyEnd = gemmaBalancedBodyEnd(in: afterStart, from: bodyStart) else {
                visible += remainder[startRange.lowerBound...]
                remainder = Substring("")
                break
            }
            let body = String(afterStart[bodyStart ..< bodyEnd])
            var rest = afterStart[afterStart.index(after: bodyEnd)...]
            if let terminator = rest.range(of: "<tool_call|>"),
                rest[..<terminator.lowerBound].allSatisfy({ $0.isWhitespace })
            {
                rest = rest[terminator.upperBound...]
            }
            var parser = LlamaGemmaArgumentParser(body)
            if !name.isEmpty, let argumentsJSON = parser.parseObjectJSON() {
                calls.append(LlamaParsedToolCall(name: name, argumentsJSON: argumentsJSON))
            }
            remainder = rest
        }
        visible += remainder
        let stripped = Self.stripGemmaThoughtChannels(from: visible)
        return (stripped.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    /// Finds the closing brace of a Gemma call body, skipping braces inside
    /// `<|"|>`-quoted strings and counting nested structures.
    private func gemmaBalancedBodyEnd(
        in text: Substring,
        from start: Substring.Index
    ) -> Substring.Index? {
        var depth = 0
        var index = start
        while index < text.endIndex {
            if text[index...].hasPrefix("<|\"|>") {
                let afterQuote = text.index(index, offsetBy: 5)
                guard let closeQuote = text[afterQuote...].range(of: "<|\"|>") else { return nil }
                index = closeQuote.upperBound
                continue
            }
            let character = text[index]
            if character == "{" || character == "[" {
                depth += 1
            } else if character == "]" {
                depth -= 1
            } else if character == "}" {
                if depth == 0 { return index }
                depth -= 1
            }
            index = text.index(after: index)
        }
        return nil
    }
}

/// Parses Gemma 4 argument notation into canonical JSON: unquoted keys,
/// `<|"|>`-quoted strings, nested objects and arrays, and bare
/// number/boolean/null literals.
struct LlamaGemmaArgumentParser {
    private let characters: [Character]
    private var index = 0

    init(_ text: String) {
        self.characters = Array(text)
    }

    /// Parses the full input as an object body (`key:value,...`) and returns
    /// it as a JSON object string, or `nil` if the input is malformed.
    mutating func parseObjectJSON() -> String? {
        guard let object = parseObjectBody(terminators: []) else { return nil }
        guard
            let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            return nil
        }
        skipWhitespace()
        guard index >= characters.count else { return nil }
        return json
    }

    private mutating func parseObjectBody(terminators: Set<Character>) -> [String: Any]? {
        var object: [String: Any] = [:]
        skipWhitespace()
        while index < characters.count, !terminators.contains(characters[index]) {
            guard let key = parseKey() else { return nil }
            guard consume(":") else { return nil }
            guard let value = parseValue() else { return nil }
            object[key] = value
            skipWhitespace()
            if index < characters.count, characters[index] == "," {
                index += 1
                skipWhitespace()
            } else {
                break
            }
        }
        return object
    }

    private mutating func parseKey() -> String? {
        skipWhitespace()
        if let quoted = parseQuotedString() { return quoted }
        var key = ""
        while index < characters.count {
            let character = characters[index]
            if character == ":" || character == "," || character == "}" { break }
            key.append(character)
            index += 1
        }
        let trimmed = key.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? nil : trimmed
    }

    private mutating func parseValue() -> Any? {
        skipWhitespace()
        if let string = parseQuotedString() { return string }
        guard index < characters.count else { return nil }
        switch characters[index] {
        case "{":
            index += 1
            guard let object = parseObjectBody(terminators: ["}"]) else { return nil }
            guard consume("}") else { return nil }
            return object
        case "[":
            index += 1
            var array: [Any] = []
            skipWhitespace()
            while index < characters.count, characters[index] != "]" {
                guard let element = parseValue() else { return nil }
                array.append(element)
                skipWhitespace()
                if index < characters.count, characters[index] == "," {
                    index += 1
                    skipWhitespace()
                }
            }
            guard consume("]") else { return nil }
            return array
        default:
            var literal = ""
            while index < characters.count {
                let character = characters[index]
                if character == "," || character == "}" || character == "]" { break }
                literal.append(character)
                index += 1
            }
            let trimmed = literal.trimmingCharacters(in: .whitespaces)
            switch trimmed {
            case "true": return true
            case "false": return false
            case "null": return NSNull()
            default:
                if let integer = Int64(trimmed) { return integer }
                if let double = Double(trimmed) { return double }
                return trimmed
            }
        }
    }

    private mutating func parseQuotedString() -> String? {
        guard remainingHasPrefix("<|\"|>") else { return nil }
        index += 5
        var value = ""
        while index < characters.count {
            if remainingHasPrefix("<|\"|>") {
                index += 5
                return value
            }
            value.append(characters[index])
            index += 1
        }
        return nil
    }

    private func remainingHasPrefix(_ prefix: String) -> Bool {
        let prefixCharacters = Array(prefix)
        guard index + prefixCharacters.count <= characters.count else { return false }
        for offset in 0 ..< prefixCharacters.count
        where characters[index + offset] != prefixCharacters[offset] {
            return false
        }
        return true
    }

    private mutating func skipWhitespace() {
        while index < characters.count, characters[index].isWhitespace {
            index += 1
        }
    }

    private mutating func consume(_ character: Character) -> Bool {
        skipWhitespace()
        guard index < characters.count, characters[index] == character else { return false }
        index += 1
        return true
    }
}

/// Whether an `NSNumber` produced by JSON decoding holds a boolean.
///
/// Core Foundation type identity is the exact check on Darwin. swift-corelibs-foundation
/// has no `CFBoolean`, so other platforms fall back to the encoded Objective-C type,
/// which JSON decoding sets to `c` only for booleans.
func isBooleanNumber(_ number: NSNumber) -> Bool {
    #if canImport(Darwin)
        return CFGetTypeID(number) == CFBooleanGetTypeID()
    #else
        return String(cString: number.objCType) == "c"
    #endif
}
