import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    private struct LlamaNestedArgumentsTool: Tool {
        let name = "find_weather"
        let description = "Find weather for a location."

        @Generable
        enum Units {
            case celsius
            case fahrenheit
        }

        @Generable
        struct Location {
            var city: String
            var units: Units
        }

        @Generable
        struct Arguments {
            var location: Location
            var alternatives: [Location]
        }

        func call(arguments: Arguments) async throws -> String {
            arguments.location.city
        }
    }

    private struct LlamaSchemaOptOutTool: Tool {
        let name = "innate_weather"
        let description = "Weather lookup known to the model."
        let includesSchemaInInstructions = false

        var parameters: GenerationSchema {
            Issue.record("An opted-out tool's schema should not be read for the prompt.")
            return WeatherTool.Arguments.generationSchema
        }

        func call(arguments: WeatherTool.Arguments) async throws -> String {
            arguments.city
        }
    }

    @Suite("LlamaToolCallFormat")
    struct LlamaToolCallFormatTests {
        private let weatherTool = LlamaToolDefinition(
            name: "get_weather",
            description: "Get the current weather for a city",
            parameters: [
                "type": "object",
                "properties": [
                    "city": [
                        "type": "string",
                        "description": "The city name",
                    ]
                ],
                "required": ["city"],
            ]
        )

        // MARK: - Detection

        @Test func detectsGemmaFromTurnMarker() {
            let template = "{{- '<|turn>' + role + '\\n' }}"
            #expect(LlamaToolCallFormat.detect(template: template) == .gemma)
        }

        @Test func detectsQwenXMLFromFunctionMarker() {
            let template = "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}"
            #expect(LlamaToolCallFormat.detect(template: template) == .qwenXML)
        }

        @Test func defaultsToHermesJSON() {
            #expect(LlamaToolCallFormat.detect(template: "<|im_start|>{{ role }}") == .hermesJSON)
            #expect(LlamaToolCallFormat.detect(template: nil) == .hermesJSON)
        }

        // MARK: - System prompt rendering

        @Test func hermesSystemMessageWrapsToolSpecs() throws {
            let message = try LlamaToolCallFormat.hermesJSON.systemMessage(
                existingText: "You are helpful.",
                tools: [weatherTool]
            )
            #expect(message.hasPrefix("You are helpful.\n\n# Tools"))
            #expect(message.contains("<tools>"))
            #expect(message.contains("\"name\":\"get_weather\""))
            #expect(message.contains("{\"name\": <function-name>, \"arguments\": <args-json-object>}"))
        }

        @Test func qwenXMLSystemMessagePutsToolsFirst() throws {
            let message = try LlamaToolCallFormat.qwenXML.systemMessage(
                existingText: "You are helpful.",
                tools: [weatherTool]
            )
            #expect(message.hasPrefix("# Tools"))
            #expect(message.hasSuffix("You are helpful."))
            #expect(message.contains("<function=example_function_name>"))
        }

        @Test func gemmaSystemMessageAppendsDeclarations() throws {
            let message = try LlamaToolCallFormat.gemma.systemMessage(
                existingText: "You are helpful.",
                tools: [weatherTool]
            )
            #expect(message.hasPrefix("You are helpful.<|tool>declaration:get_weather{"))
            #expect(message.hasSuffix("<tool|>"))
            #expect(message.contains("description:<|\"|>Get the current weather for a city<|\"|>"))
            #expect(message.contains("city:{description:<|\"|>The city name<|\"|>,type:<|\"|>STRING<|\"|>}"))
            #expect(message.contains("required:[<|\"|>city<|\"|>]"))
            #expect(message.contains("type:<|\"|>OBJECT<|\"|>"))
        }

        @Test func emptyToolListLeavesSystemTextUntouched() throws {
            let message = try LlamaToolCallFormat.hermesJSON.systemMessage(existingText: "Hi.", tools: [])
            #expect(message == "Hi.")
        }

        @Test(
            arguments: [LlamaToolCallFormat.hermesJSON, .qwenXML, .gemma],
            [false, true]
        )
        func promptDefinitionsHonorSchemaOptOut(
            format: LlamaToolCallFormat,
            includeAdvertisedTool: Bool
        ) throws {
            let optedOutTool = LlamaSchemaOptOutTool()
            var tools: [any Tool] = [optedOutTool]
            if includeAdvertisedTool {
                tools.append(WeatherTool())
            }
            let context = try LlamaLanguageModel.LlamaToolPromptContext(format: format, tools: tools)
            #expect(context.definitions.map(\.name) == (includeAdvertisedTool ? ["getWeather"] : []))
            let message = try context.format.systemMessage(existingText: "Hi.", tools: context.definitions)
            #expect(!message.contains(optedOutTool.name))
            #expect(!message.contains(optedOutTool.description))
            if includeAdvertisedTool {
                #expect(message.contains("getWeather"))
                #expect(message.contains("city"))
            } else {
                #expect(message == "Hi.")
            }
        }

        @Test func gemmaResolvesNestedGenerableToolArguments() throws {
            let context = try LlamaLanguageModel.LlamaToolPromptContext(
                format: .gemma,
                tools: [LlamaNestedArgumentsTool()]
            )
            let message = try context.format.systemMessage(existingText: "", tools: context.definitions)
            #expect(!message.contains("$ref"))
            #expect(!message.contains("$defs"))
            #expect(
                message.contains(
                    "location:{description:<|\"|>Generated Location<|\"|>,properties:{city:{type:<|\"|>STRING<|\"|>}"
                )
            )
            #expect(
                message.contains(
                    "alternatives:{items:{additionalProperties:false,description:<|\"|>Generated Location<|\"|>,properties:{city:"
                )
            )
            #expect(
                message.contains(
                    "units:{description:<|\"|>Generated Units<|\"|>,enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}"
                )
            )
            #expect(message.components(separatedBy: "type:<|\"|>OBJECT<|\"|>").count == 4)
        }

        @Test func gemmaResolvesRootAndChainedEnumReferences() throws {
            let tool = LlamaToolDefinition(
                name: "f",
                description: "",
                parameters: [
                    "$ref": "#/$defs/Arguments",
                    "$defs": [
                        "Arguments": [
                            "type": "object",
                            "properties": [
                                "units": ["$ref": "#/$defs/Alias", "description": "Preferred units"],
                                "choices": ["type": "array", "items": ["$ref": "#/$defs/Units~1~0"]],
                                "groups": [
                                    "type": "array",
                                    "items": ["type": "array", "items": ["$ref": "#/$defs/Units~1~0"]],
                                ],
                            ],
                        ],
                        "Alias": ["$ref": "#/$defs/Units~1~0"],
                        "Units/~": ["type": "string", "enum": ["celsius", "fahrenheit"]],
                    ],
                ]
            )
            let message = try LlamaToolCallFormat.gemma.systemMessage(existingText: "", tools: [tool])
            let enumFields = "enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>"
            #expect(message.contains("units:{description:<|\"|>Preferred units<|\"|>,\(enumFields)}"))
            #expect(message.contains("choices:{items:{\(enumFields)},type:<|\"|>ARRAY<|\"|>}"))
            #expect(
                message.contains("groups:{items:{items:{\(enumFields)},type:<|\"|>ARRAY<|\"|>},type:<|\"|>ARRAY<|\"|>}")
            )
            #expect(!message.contains("$ref"))
        }

        @Test func gemmaRejectsUnresolvedSchemaReferences() {
            let tool = LlamaToolDefinition(name: "f", description: "", parameters: ["$ref": "#/$defs/Missing"])
            #expect(throws: LlamaToolCallFormat.SchemaRenderingError.unresolvedReference("#/$defs/Missing")) {
                try LlamaToolCallFormat.gemma.systemMessage(existingText: "", tools: [tool])
            }
        }

        @Test func gemmaRejectsRecursiveSchemaReferences() {
            let tool = LlamaToolDefinition(
                name: "f",
                description: "",
                parameters: [
                    "$ref": "#/$defs/Node",
                    "$defs": [
                        "Node": [
                            "type": "object",
                            "properties": ["children": ["type": "array", "items": ["$ref": "#/$defs/Node"]]],
                        ]
                    ],
                ]
            )
            #expect(throws: LlamaToolCallFormat.SchemaRenderingError.recursiveReference("#/$defs/Node")) {
                try LlamaToolCallFormat.gemma.systemMessage(existingText: "", tools: [tool])
            }
        }

        enum SchemaPosition: CaseIterable {
            case root, property, arrayItem

            func wrap(_ schema: DynamicGenerationSchema) -> DynamicGenerationSchema {
                switch self {
                case .root:
                    return schema
                case .property:
                    return DynamicGenerationSchema(
                        name: "Arguments",
                        properties: [.init(name: "value", schema: schema)]
                    )
                case .arrayItem:
                    return DynamicGenerationSchema(
                        name: "Arguments",
                        properties: [.init(name: "values", schema: .init(arrayOf: schema))]
                    )
                }
            }

            func wrap(_ schema: [String: Any]) -> [String: Any] {
                switch self {
                case .root: return schema
                case .property: return ["type": "object", "properties": ["value": schema]]
                case .arrayItem:
                    return ["type": "object", "properties": ["values": ["type": "array", "items": schema]]]
                }
            }
        }

        @Test(arguments: SchemaPosition.allCases)
        func gemmaRejectsReferencedDynamicUnions(position: SchemaPosition) throws {
            let choice = DynamicGenerationSchema(
                name: "Choice",
                anyOf: [.init(type: String.self), .init(type: Int.self)]
            )
            let schema = try GenerationSchema(root: position.wrap(choice), dependencies: [])
            let data = try JSONEncoder().encode(schema)
            let parameters = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let tool = LlamaToolDefinition(name: "f", description: "", parameters: parameters)
            #expect(throws: LlamaToolCallFormat.SchemaRenderingError.unsupportedComposition("anyOf")) {
                try LlamaToolCallFormat.gemma.systemMessage(existingText: "", tools: [tool])
            }
        }

        @Test(arguments: SchemaPosition.allCases, ["anyOf", "oneOf", "allOf"])
        func gemmaRejectsInlineSchemaCompositions(position: SchemaPosition, keyword: String) {
            let parameters = position.wrap([keyword: [["type": "string"], ["type": "integer"]]])
            let tool = LlamaToolDefinition(name: "f", description: "", parameters: parameters)
            #expect(throws: LlamaToolCallFormat.SchemaRenderingError.unsupportedComposition(keyword)) {
                try LlamaToolCallFormat.gemma.systemMessage(existingText: "", tools: [tool])
            }
        }

        @Test(arguments: [LlamaToolCallFormat.hermesJSON, .qwenXML])
        func otherFormatsPreserveUnionSchemas(format: LlamaToolCallFormat) throws {
            let tool = LlamaToolDefinition(
                name: "f",
                description: "",
                parameters: ["anyOf": [["type": "string"], ["type": "integer"]]]
            )
            let message = try format.systemMessage(existingText: "", tools: [tool])
            #expect(message.contains("\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]"))
        }

        // MARK: - Hermes JSON parsing

        @Test func parsesHermesCall() {
            let text = """
                Let me check that for you.
                <tool_call>
                {"name": "get_weather", "arguments": {"city": "Paris"}}
                </tool_call>
                """
            let (visible, calls) = LlamaToolCallFormat.hermesJSON.parseToolCalls(in: text)
            #expect(visible == "Let me check that for you.\n")
            #expect(calls.count == 1)
            #expect(calls.first?.name == "get_weather")
            #expect(calls.first?.argumentsJSON == "{\"city\":\"Paris\"}")
        }

        @Test func parsesHermesCallWithStringEncodedArguments() {
            let text = "<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>"
            let (_, calls) = LlamaToolCallFormat.hermesJSON.parseToolCalls(in: text)
            #expect(calls.first?.argumentsJSON == "{\"a\": 1}")
        }

        @Test func parsesMultipleHermesCalls() {
            let text = """
                <tool_call>
                {"name": "a", "arguments": {}}
                </tool_call>
                <tool_call>
                {"name": "b", "arguments": {"x": 2}}
                </tool_call>
                """
            let (visible, calls) = LlamaToolCallFormat.hermesJSON.parseToolCalls(in: text)
            #expect(visible == "\n")
            #expect(calls.map(\.name) == ["a", "b"])
        }

        @Test func plainTextHasNoHermesCalls() {
            let (visible, calls) = LlamaToolCallFormat.hermesJSON.parseToolCalls(in: "Just an answer.")
            #expect(visible == "Just an answer.")
            #expect(calls.isEmpty)
        }

        @Test func unterminatedHermesBlockStaysVisible() {
            let text = "Answer <tool_call>{\"name\": \"a\""
            let (visible, calls) = LlamaToolCallFormat.hermesJSON.parseToolCalls(in: text)
            #expect(calls.isEmpty)
            #expect(visible.contains("<tool_call>"))
        }

        // MARK: - Qwen XML parsing

        @Test func parsesQwenXMLCall() {
            let text = """
                I will look that up.
                <tool_call>
                <function=get_weather>
                <parameter=city>
                Paris
                </parameter>
                </function>
                </tool_call>
                """
            let (visible, calls) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: text)
            #expect(visible == "I will look that up.\n")
            #expect(calls.count == 1)
            #expect(calls.first?.name == "get_weather")
            #expect(calls.first?.argumentsJSON == "{\"city\":\"Paris\"}")
        }

        @Test func qwenXMLPreservesMultilineParameterValues() {
            let text = """
                <tool_call>
                <function=save_note>
                <parameter=body>
                line one
                line two
                </parameter>
                </function>
                </tool_call>
                """
            let (_, calls) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: text)
            #expect(calls.first?.argumentsJSON == "{\"body\":\"line one\\nline two\"}")
        }

        @Test func qwenXMLDecodesStructuredParameterValues() {
            let text = """
                <tool_call>
                <function=f>
                <parameter=items>
                ["a", "b"]
                </parameter>
                </function>
                </tool_call>
                """
            let (_, calls) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: text)
            #expect(calls.first?.argumentsJSON == "{\"items\":[\"a\",\"b\"]}")
        }

        @Test func qwenXMLAcceptsCompleteZeroArgumentCall() {
            let text = "<tool_call><function=f></function></tool_call>"
            let (_, calls) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: text)
            #expect(calls == [LlamaParsedToolCall(name: "f", argumentsJSON: "{}")])
        }

        @Test func qwenXMLAcceptsMultipleCompleteParameters() {
            let text = """
                <tool_call>
                <function=f>
                <parameter=city>Paris</parameter>
                <parameter=units>celsius</parameter>
                </function>
                </tool_call>
                """
            let (_, calls) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: text)
            #expect(calls.first?.argumentsJSON == "{\"city\":\"Paris\",\"units\":\"celsius\"}")
        }

        @Test(arguments: [
            "<function=side_effect>",
            "<function=f><parameter=city>Paris</parameter>",
            "<function=f><parameter=city>Paris</function>",
            "<function=f><parameter=city</function>",
            "<function=f><parameter=city>Paris</parameter><parameter=units</function>",
            "<function=f><parameter=city>Paris</parameter><parameter=units>celsius</function>",
            "<function=f><parameter=city>Paris<parameter=units>celsius</parameter></function>",
            "<function=f><parameter>Paris</parameter></function>",
            "<function=f><parameter=>Paris</parameter></function>",
            "<function=f><parameter=city</parameter></function>",
            "<function=f><parameter=city>Paris</function></parameter>",
            "<function=f></function><parameter=city>Paris</parameter>",
        ])
        func qwenXMLRejectsIncompleteOrMalformedCalls(body: String) {
            let (_, calls) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: "<tool_call>\(body)</tool_call>")
            #expect(calls.isEmpty)
        }

        // MARK: - Gemma parsing

        @Test func parsesGemmaCall() {
            let text = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
            let (visible, calls) = LlamaToolCallFormat.gemma.parseToolCalls(in: text)
            #expect(visible.isEmpty)
            #expect(calls.count == 1)
            #expect(calls.first?.name == "get_weather")
            #expect(calls.first?.argumentsJSON == "{\"city\":\"Paris\"}")
        }

        @Test func gemmaQuotedStringsMayContainStructuralCharacters() {
            let text = "<|tool_call>call:f{note:<|\"|>a, {b}: [c]<|\"|>}<tool_call|>"
            let (_, calls) = LlamaToolCallFormat.gemma.parseToolCalls(in: text)
            #expect(calls.first?.argumentsJSON == "{\"note\":\"a, {b}: [c]\"}")
        }

        @Test func gemmaParsesScalarAndNestedArguments() {
            let text =
                "<|tool_call>call:f{count:3,enabled:true,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],meta:{k:<|\"|>v<|\"|>}}<tool_call|>"
            let (_, calls) = LlamaToolCallFormat.gemma.parseToolCalls(in: text)
            #expect(
                calls.first?.argumentsJSON
                    == "{\"count\":3,\"enabled\":true,\"meta\":{\"k\":\"v\"},\"tags\":[\"a\",\"b\"]}"
            )
        }

        @Test func gemmaCallWithoutTerminatorStillParses() {
            let text = "<|tool_call>call:f{city:<|\"|>Paris<|\"|>}"
            let (_, calls) = LlamaToolCallFormat.gemma.parseToolCalls(in: text)
            #expect(calls.first?.name == "f")
        }

        // MARK: - Transcript replay round trips

        @Test func hermesAssistantTextRoundTrips() {
            let call = LlamaParsedToolCall(name: "get_weather", argumentsJSON: "{\"city\":\"Paris\"}")
            let text = LlamaToolCallFormat.hermesJSON.assistantText(for: [call], precededByContent: false)
            let (_, parsed) = LlamaToolCallFormat.hermesJSON.parseToolCalls(in: text)
            #expect(parsed == [call])
        }

        @Test func qwenXMLAssistantTextRoundTrips() {
            let call = LlamaParsedToolCall(name: "get_weather", argumentsJSON: "{\"city\":\"Paris\"}")
            let text = LlamaToolCallFormat.qwenXML.assistantText(for: [call], precededByContent: false)
            let (_, parsed) = LlamaToolCallFormat.qwenXML.parseToolCalls(in: text)
            #expect(parsed == [call])
        }

        @Test func gemmaAssistantTextRoundTrips() {
            let call = LlamaParsedToolCall(name: "get_weather", argumentsJSON: "{\"city\":\"Paris\"}")
            let text = LlamaToolCallFormat.gemma.assistantText(for: [call], precededByContent: false)
            #expect(text == "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
            let (_, parsed) = LlamaToolCallFormat.gemma.parseToolCalls(in: text)
            #expect(parsed == [call])
        }

        // MARK: - Gemma thought channels

        @Test func stripsCompletedThoughtChannels() {
            let text = "<|channel>thought\nThe user said hi.\n<channel|>Hello there!"
            #expect(LlamaToolCallFormat.stripGemmaThoughtChannels(from: text) == "Hello there!")
        }

        @Test func stripsAlternateChannelSpelling() {
            let text = "<|channel|>thought\nReasoning.\n<channel|>Answer."
            #expect(LlamaToolCallFormat.stripGemmaThoughtChannels(from: text) == "Answer.")
        }

        @Test func stripsUnclosedThoughtChannelToEnd() {
            let text = "Partial<|channel>thought\nstill thinking"
            #expect(LlamaToolCallFormat.stripGemmaThoughtChannels(from: text) == "Partial")
        }

        @Test func stripsMultipleThoughtChannels() {
            let text = "<|channel>thought\na\n<channel|>X<|channel>thought\nb\n<channel|>Y"
            #expect(LlamaToolCallFormat.stripGemmaThoughtChannels(from: text) == "XY")
        }

        @Test func gemmaParseStripsThoughtChannels() {
            let text = "<|channel>thought\nplan\n<channel|>Done.<|tool_call>call:f{}<tool_call|>"
            let (visible, calls) = LlamaToolCallFormat.gemma.parseToolCalls(in: text)
            #expect(visible == "Done.")
            #expect(calls.count == 1)
        }

        // MARK: - Streaming visibility

        @Test(
            arguments: LlamaToolCallFormat.gemmaChannelOpenMarkers,
            ["\n  Hello.\n", "  \t\n", " Hi <"]
        )
        func gemmaCompletedResponsePreservesWhitespace(channelMarker: String, answer: String) {
            let raw = " \n" + channelMarker + "thought\nplan\n<channel|>" + answer
            let visible = LlamaToolCallFormat.gemma.streamingVisibleText(
                in: raw,
                withholdToolCalls: false,
                holdPartialMarkers: false
            )
            #expect(visible == " \n" + answer)
            #expect(visible == LlamaToolCallFormat.gemma.parseToolCalls(in: raw).visibleText)
        }

        @Test(
            arguments: [LlamaToolCallFormat.hermesJSON, .qwenXML, .gemma],
            [" ", "\n", "\n\n", "\t"]
        )
        func preservesWhitespaceBetweenToolRounds(format: LlamaToolCallFormat, separator: String) {
            let markup = format.assistantText(
                for: [LlamaParsedToolCall(name: "get_weather", argumentsJSON: "{\"city\":\"Paris\"}")],
                precededByContent: false
            )
            let rounds = ["Checking." + separator + markup, markup, "It is 72."]
            let response = rounds.map { format.parseToolCalls(in: $0).visibleText }.joined()
            let streamed = rounds.map {
                format.streamingVisibleText(in: $0, withholdToolCalls: true, holdPartialMarkers: false)
            }.joined()
            #expect(response == "Checking." + separator + "It is 72.")
            #expect(response == streamed)
        }

        @Test(arguments: [LlamaToolCallFormat.hermesJSON, .qwenXML, .gemma])
        func preservesWhitespaceInPlainText(format: LlamaToolCallFormat) {
            let text = "\n  It is 72.\n"
            let (visible, calls) = format.parseToolCalls(in: text)
            #expect(visible == text)
            #expect(calls.isEmpty)
        }

        @Test func streamingWithholdsPartialToolCallMarker() {
            let visible = LlamaToolCallFormat.hermesJSON.streamingVisibleText(
                in: "The answer is<tool_",
                withholdToolCalls: true
            )
            #expect(visible == "The answer is")
        }

        @Test func streamingFlushesBrokenMarkerPrefix() {
            let visible = LlamaToolCallFormat.hermesJSON.streamingVisibleText(
                in: "5 < 10 is true",
                withholdToolCalls: true
            )
            #expect(visible == "5 < 10 is true")
        }

        @Test func streamingReleasesPartialMarkerAtEndOfRound() {
            let visible = LlamaToolCallFormat.hermesJSON.streamingVisibleText(
                in: "The answer is <",
                withholdToolCalls: true,
                holdPartialMarkers: false
            )
            #expect(visible == "The answer is <")
        }

        @Test func streamingTruncatesAtCompleteToolCallStart() {
            let visible = LlamaToolCallFormat.hermesJSON.streamingVisibleText(
                in: "Checking.<tool_call>\n{\"name\":",
                withholdToolCalls: true
            )
            #expect(visible == "Checking.")
        }

        @Test func streamingIgnoresToolMarkersWhenToolsInactive() {
            let visible = LlamaToolCallFormat.hermesJSON.streamingVisibleText(
                in: "text <tool_call> more",
                withholdToolCalls: false
            )
            #expect(visible == "text <tool_call> more")
        }

        @Test func streamingWithholdsGemmaThoughtChannel() {
            let format = LlamaToolCallFormat.gemma
            #expect(format.streamingVisibleText(in: "<|chan", withholdToolCalls: false) == "")
            #expect(
                format.streamingVisibleText(
                    in: "<|channel>thought\nhmm",
                    withholdToolCalls: false
                ) == ""
            )
            #expect(
                format.streamingVisibleText(
                    in: "<|channel>thought\nhmm\n<channel|>Hi",
                    withholdToolCalls: false
                ) == "Hi"
            )
        }

        @Test func streamingWithholdsGemmaPartialToolMarkerAfterThought() {
            let format = LlamaToolCallFormat.gemma
            let raw = "<|channel>thought\nplan\n<channel|>Sure.<|tool_"
            #expect(format.streamingVisibleText(in: raw, withholdToolCalls: true) == "Sure.")
        }

        // MARK: - Tool response messages

        @Test func hermesToolResponseIsAUserTurn() {
            let message = LlamaToolCallFormat.hermesJSON.toolResponseMessage(
                toolName: "get_weather",
                segments: [.text(.init(content: "{\"temperature\": 21}"))]
            )
            #expect(message.role == "user")
            #expect(message.content == "<tool_response>\n{\"temperature\": 21}\n</tool_response>")
        }

        @Test func gemmaToolResponseContinuesTheModelTurn() throws {
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(
                toolName: "get_weather",
                segments: [
                    .structure(
                        .init(source: "get_weather", content: try GeneratedContent(json: "{\"temperature\":21}"))
                    )
                ]
            )
            #expect(message.role == "tool")
            #expect(
                message.content
                    == "<|tool_response>response:get_weather{temperature:21}<tool_response|>"
            )
        }

        @Test func gemmaScalarToolResponseWrapsInValue() {
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(
                toolName: "f",
                segments: [.text(.init(content: "done"))]
            )
            #expect(message.content == "<|tool_response>response:f{value:<|\"|>done<|\"|>}<tool_response|>")
        }

        @Test(arguments: [
            ("[\"a\"]", "{value:[<|\"|>a<|\"|>]}"),
            ("3", "{value:3}"),
            ("1.5", "{value:1.5}"),
            ("true", "{value:true}"),
            ("false", "{value:false}"),
            ("null", "{value:null}"),
            ("\"done\"", "{value:<|\"|>done<|\"|>}"),
            ("[]", "{value:[]}"),
            ("{}", "{}"),
            ("{\"items\":[3,true,null]}", "{items:[3,true,null]}"),
        ])
        func gemmaStructuredToolResponsePreservesType(testCase: (String, String)) throws {
            let (json, body) = testCase
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(
                toolName: "f",
                segments: [.structure(.init(source: "f", content: try GeneratedContent(json: json)))]
            )
            #expect(message.content == "<|tool_response>response:f\(body)<tool_response|>")
        }

        @Test(arguments: ["[\"a\"]", "3", "true", "null", "{\"items\":[3]}", "\"done\""])
        func gemmaTextToolResponseStaysText(text: String) {
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(
                toolName: "f",
                segments: [.text(.init(content: text))]
            )
            #expect(message.content == "<|tool_response>response:f{value:<|\"|>\(text)<|\"|>}<tool_response|>")
        }

        @Test func gemmaMixedToolResponsePreservesSegmentTypesAndOrder() throws {
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(
                toolName: "f",
                segments: [
                    .text(.init(content: "Count:")),
                    .structure(.init(source: "f", content: GeneratedContent(3))),
                    .structure(.init(source: "f", content: try GeneratedContent(json: "[true]"))),
                ]
            )
            #expect(
                message.content == "<|tool_response>response:f{value:[<|\"|>Count:<|\"|>,3,[true]]}<tool_response|>"
            )
        }

        @Test(arguments: [LlamaToolCallFormat.hermesJSON, .qwenXML])
        func textToolResponseFormatsStillJoinSegments(format: LlamaToolCallFormat) {
            let message = format.toolResponseMessage(
                toolName: "f",
                segments: [
                    .text(.init(content: "Count:")),
                    .structure(.init(source: "f", content: GeneratedContent(3))),
                ]
            )
            #expect(message.role == "user")
            #expect(message.content == "<tool_response>\nCount:\n3\n</tool_response>")
        }
    }

    @Suite(
        "LlamaLanguageModel tools",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_TOOL_MODEL_PATH"] != nil)
    )
    struct LlamaLanguageModelToolTests {
        let model = LlamaLanguageModel(
            modelPath: ProcessInfo.processInfo.environment["LLAMA_TOOL_MODEL_PATH"]!
        )

        @Test func executesToolAndAnswersFromItsOutput() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            var options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 1024)
            options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
            let response = try await session.respond(
                to: "How's the weather in Paris? Use the getWeather tool.",
                options: options
            )

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.toolName == weatherTool.name)
                foundToolOutput = true
            }
            #expect(foundToolOutput)

            let calls = await weatherTool.calls
            #expect(calls.count == 1)
            #expect(calls.first?.arguments.city.contains("Paris") == true)
            #expect(response.content.lowercased().contains("72") || response.content.lowercased().contains("sunny"))
            #expect(!response.content.contains("<tool_call>"))
        }

        @Test func replaysToolExchangeInFollowUpTurns() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            var options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 1024)
            options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
            _ = try await session.respond(
                to: "How's the weather in Paris? Use the getWeather tool.",
                options: options
            )
            let followUp = try await session.respond(
                to: "What temperature did you just report, in Fahrenheit? Answer with just the number.",
                options: options
            )

            let calls = await weatherTool.calls
            #expect(calls.count == 1)
            #expect(followUp.content.contains("72"))
        }

        @Test func streamsToolExchangeProgressively() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            var options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 1024)
            options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
            let stream = session.streamResponse(
                to: "How's the weather in Paris? Use the getWeather tool.",
                options: options
            )
            var snapshots: [String] = []
            var sawToolOutputEntry = false
            for try await snapshot in stream {
                snapshots.append(snapshot.content)
                for case .toolOutput(_) in snapshot.transcriptEntries {
                    sawToolOutputEntry = true
                }
            }

            let calls = await weatherTool.calls
            #expect(calls.count == 1)
            #expect(sawToolOutputEntry)
            #expect(snapshots.count > 3)
            let final = snapshots.last ?? ""
            #expect(final.lowercased().contains("72") || final.lowercased().contains("sunny"))
            for content in snapshots {
                #expect(!content.contains("<tool_call>"))
                #expect(!content.contains("<|tool_call>"))
                #expect(!content.contains("<|channel"))
            }
        }

        @Test func answersDirectlyWhenNoToolApplies() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            var options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 1024)
            options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
            let response = try await session.respond(
                to: "What is 2 + 2? Answer with just the number.",
                options: options
            )

            let calls = await weatherTool.calls
            #expect(calls.isEmpty)
            #expect(response.content.contains("4"))
        }
    }
#endif
