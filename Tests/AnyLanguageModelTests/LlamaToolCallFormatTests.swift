import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
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

        @Test func hermesSystemMessageWrapsToolSpecs() {
            let message = LlamaToolCallFormat.hermesJSON.systemMessage(
                existingText: "You are helpful.",
                tools: [weatherTool]
            )
            #expect(message.hasPrefix("You are helpful.\n\n# Tools"))
            #expect(message.contains("<tools>"))
            #expect(message.contains("\"name\":\"get_weather\""))
            #expect(message.contains("{\"name\": <function-name>, \"arguments\": <args-json-object>}"))
        }

        @Test func qwenXMLSystemMessagePutsToolsFirst() {
            let message = LlamaToolCallFormat.qwenXML.systemMessage(
                existingText: "You are helpful.",
                tools: [weatherTool]
            )
            #expect(message.hasPrefix("# Tools"))
            #expect(message.hasSuffix("You are helpful."))
            #expect(message.contains("<function=example_function_name>"))
        }

        @Test func gemmaSystemMessageAppendsDeclarations() {
            let message = LlamaToolCallFormat.gemma.systemMessage(
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

        @Test func emptyToolListLeavesSystemTextUntouched() {
            let message = LlamaToolCallFormat.hermesJSON.systemMessage(existingText: "Hi.", tools: [])
            #expect(message == "Hi.")
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
            #expect(visible == "Let me check that for you.")
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
            #expect(visible.isEmpty)
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
            #expect(visible == "I will look that up.")
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

        // MARK: - Tool response messages

        @Test func hermesToolResponseIsAUserTurn() {
            let message = LlamaToolCallFormat.hermesJSON.toolResponseMessage(
                toolName: "get_weather",
                content: "{\"temperature\": 21}"
            )
            #expect(message.role == "user")
            #expect(message.content == "<tool_response>\n{\"temperature\": 21}\n</tool_response>")
        }

        @Test func gemmaToolResponseContinuesTheModelTurn() {
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(
                toolName: "get_weather",
                content: "{\"temperature\": 21}"
            )
            #expect(message.role == "tool")
            #expect(
                message.content
                    == "<|tool_response>response:get_weather{temperature:21}<tool_response|>"
            )
        }

        @Test func gemmaScalarToolResponseWrapsInValue() {
            let message = LlamaToolCallFormat.gemma.toolResponseMessage(toolName: "f", content: "done")
            #expect(message.content == "<|tool_response>response:f{value:<|\"|>done<|\"|>}<tool_response|>")
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
