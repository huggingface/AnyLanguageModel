import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("CustomGenerationOptions")
struct CustomGenerationOptionsTests {

    // MARK: - Protocol Conformance

    @Test func neverConformsToCustomGenerationOptions() {
        // Never should conform to CustomGenerationOptions (used as default)
        let _: any CustomGenerationOptions.Type = Never.self
    }

    // MARK: - Subscript Access

    @Test func subscriptGetReturnsNilWhenNotSet() {
        let options = GenerationOptions()
        let customOptions = options[custom: OpenAILanguageModel.self]
        #expect(customOptions == nil)
    }

    @Test func subscriptSetAndGet() {
        var options = GenerationOptions()
        let customOptions = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: ["test": .string("value")]
        )

        options[custom: OpenAILanguageModel.self] = customOptions

        let retrieved = options[custom: OpenAILanguageModel.self]
        #expect(retrieved != nil)
        #expect(retrieved?.extraBody?["test"] == .string("value"))
    }

    @Test func subscriptSetToNilRemovesValue() {
        var options = GenerationOptions()
        options[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])

        #expect(options[custom: OpenAILanguageModel.self] != nil)

        options[custom: OpenAILanguageModel.self] = nil

        #expect(options[custom: OpenAILanguageModel.self] == nil)
    }

    @Test func subscriptIsolatesModelTypes() {
        var options = GenerationOptions()

        // Set custom options for OpenAI
        options[custom: OpenAILanguageModel.self] = .init(
            extraBody: ["openai_key": .string("openai_value")]
        )

        // MockLanguageModel uses Never as CustomGenerationOptions (default)
        // So we can't set custom options for it - this is by design
        let mockOptions: Never? = options[custom: MockLanguageModel.self]
        #expect(mockOptions == nil)

        // OpenAI options should still be accessible
        let openaiOptions = options[custom: OpenAILanguageModel.self]
        #expect(openaiOptions?.extraBody?["openai_key"] == .string("openai_value"))
    }

    // MARK: - Equality

    @Test func equalityWithNoCustomOptions() {
        let options1 = GenerationOptions(temperature: 0.7)
        let options2 = GenerationOptions(temperature: 0.7)

        #expect(options1 == options2)
    }

    @Test func equalityWithSameCustomOptions() {
        var options1 = GenerationOptions(temperature: 0.7)
        var options2 = GenerationOptions(temperature: 0.7)

        options1[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])
        options2[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])

        #expect(options1 == options2)
    }

    @Test func inequalityWithDifferentCustomOptions() {
        var options1 = GenerationOptions(temperature: 0.7)
        var options2 = GenerationOptions(temperature: 0.7)

        options1[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])
        options2[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(false)])

        #expect(options1 != options2)
    }

    @Test func inequalityWhenOnlyOneHasCustomOptions() {
        var options1 = GenerationOptions(temperature: 0.7)
        let options2 = GenerationOptions(temperature: 0.7)

        options1[custom: OpenAILanguageModel.self] = .init(extraBody: ["key": .bool(true)])

        #expect(options1 != options2)
    }

    // MARK: - Encoding

    @Test func encodingWithCustomOptions() throws {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OpenAILanguageModel.self] = .init(
            extraBody: ["reasoning": .object(["enabled": .bool(true)])]
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(options)
        let json = String(data: data, encoding: .utf8)!

        // Verify the JSON contains the temperature
        #expect(json.contains("\"temperature\""))
        #expect(json.contains("0.8"))

        // Verify custom options type name is in the output
        #expect(json.contains("OpenAILanguageModel"))
        #expect(json.contains("CustomGenerationOptions"))
    }

    @Test func decodingLosesCustomOptions() throws {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OpenAILanguageModel.self] = .init(
            extraBody: ["key": .string("value")]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(options)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(GenerationOptions.self, from: data)

        // Standard options should be preserved
        #expect(decoded.temperature == 0.8)

        // Custom options are lost on round-trip (documented behavior)
        #expect(decoded[custom: OpenAILanguageModel.self] == nil)
    }
}

@Suite("Anthropic CustomGenerationOptions")
struct AnthropicCustomOptionsTests {
    @Test func initialization() {
        let options = AnthropicLanguageModel.CustomGenerationOptions(
            topP: 0.9,
            topK: 40,
            stopSequences: ["END", "STOP"],
            metadata: .init(userID: "user-123"),
            toolChoice: .auto,
            thinking: .init(budgetTokens: 1024),
            serviceTier: .priority,
            extraBody: ["custom_param": .string("value")]
        )

        #expect(options.topP == 0.9)
        #expect(options.topK == 40)
        #expect(options.stopSequences == ["END", "STOP"])
        #expect(options.metadata?.userID == "user-123")
        #expect(options.toolChoice == .auto)
        #expect(options.thinking?.budgetTokens == 1024)
        #expect(options.serviceTier == .priority)
        #expect(options.extraBody?["custom_param"] == .string("value"))
    }

    @Test func equality() {
        let options1 = AnthropicLanguageModel.CustomGenerationOptions(
            topP: 0.9,
            topK: 40
        )
        let options2 = AnthropicLanguageModel.CustomGenerationOptions(
            topP: 0.9,
            topK: 40
        )

        #expect(options1 == options2)
    }

    @Test func codable() throws {
        let options = AnthropicLanguageModel.CustomGenerationOptions(
            topP: 0.9,
            topK: 40,
            stopSequences: ["END"],
            metadata: .init(userID: "user-123"),
            toolChoice: .tool(name: "my_tool"),
            thinking: .init(budgetTokens: 2048),
            serviceTier: .standard
        )

        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(
            AnthropicLanguageModel.CustomGenerationOptions.self,
            from: data
        )

        #expect(decoded == options)
    }

    @Test func nilProperties() {
        let options = AnthropicLanguageModel.CustomGenerationOptions()
        #expect(options.topP == nil)
        #expect(options.topK == nil)
        #expect(options.stopSequences == nil)
        #expect(options.metadata == nil)
        #expect(options.toolChoice == nil)
        #expect(options.thinking == nil)
        #expect(options.serviceTier == nil)
        #expect(options.extraBody == nil)
    }

    @Test func integrationWithGenerationOptions() {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: AnthropicLanguageModel.self] = .init(
            topP: 0.9,
            topK: 40,
            stopSequences: ["END"],
            thinking: .init(budgetTokens: 4096)
        )

        let retrieved = options[custom: AnthropicLanguageModel.self]
        #expect(retrieved?.topP == 0.9)
        #expect(retrieved?.topK == 40)
        #expect(retrieved?.stopSequences == ["END"])
        #expect(retrieved?.thinking?.budgetTokens == 4096)
    }

    @Test func metadataCodable() throws {
        let metadata = AnthropicLanguageModel.CustomGenerationOptions.Metadata(
            userID: "user-456"
        )

        let data = try JSONEncoder().encode(metadata)
        let json = String(data: data, encoding: .utf8)!

        // Verify the JSON uses snake_case
        #expect(json.contains("user_id"))
        #expect(json.contains("user-456"))

        let decoded = try JSONDecoder().decode(
            AnthropicLanguageModel.CustomGenerationOptions.Metadata.self,
            from: data
        )
        #expect(decoded == metadata)
    }

    @Test func toolChoiceVariants() {
        let auto = AnthropicLanguageModel.CustomGenerationOptions(toolChoice: .auto)
        let any = AnthropicLanguageModel.CustomGenerationOptions(toolChoice: .any)
        let tool = AnthropicLanguageModel.CustomGenerationOptions(toolChoice: .tool(name: "search"))
        let disabled = AnthropicLanguageModel.CustomGenerationOptions(toolChoice: .disabled)

        #expect(auto.toolChoice == .auto)
        #expect(any.toolChoice == .any)
        #expect(tool.toolChoice == .tool(name: "search"))
        #expect(disabled.toolChoice == .disabled)

        // Verify they're all different
        #expect(auto != any)
        #expect(any != tool)
        #expect(tool != disabled)
    }

    @Test func toolChoiceCodable() throws {
        typealias ToolChoice = AnthropicLanguageModel.CustomGenerationOptions.ToolChoice

        let choices: [ToolChoice] = [
            .auto,
            .any,
            .tool(name: "my_tool"),
            .disabled,
        ]

        for choice in choices {
            let data = try JSONEncoder().encode(choice)
            let decoded = try JSONDecoder().decode(ToolChoice.self, from: data)
            #expect(decoded == choice)
        }
    }

    @Test func toolChoiceDisabledEncodesToNone() throws {
        let choice = AnthropicLanguageModel.CustomGenerationOptions.ToolChoice.disabled

        let data = try JSONEncoder().encode(choice)
        let json = String(data: data, encoding: .utf8)!

        // Verify it encodes to "none" for the API
        #expect(json.contains("\"none\""))
    }

    @Test func thinkingCodable() throws {
        let thinking = AnthropicLanguageModel.CustomGenerationOptions.Thinking(budgetTokens: 8192)

        let encoder = JSONEncoder()
        let data = try encoder.encode(thinking)
        let json = String(data: data, encoding: .utf8)!

        // Verify the JSON uses snake_case
        #expect(json.contains("budget_tokens"))
        #expect(json.contains("8192"))
        #expect(json.contains("enabled"))

        let decoded = try JSONDecoder().decode(
            AnthropicLanguageModel.CustomGenerationOptions.Thinking.self,
            from: data
        )
        #expect(decoded == thinking)
    }

    @Test func serviceTierValues() {
        #expect(AnthropicLanguageModel.CustomGenerationOptions.ServiceTier.auto.rawValue == "auto")
        #expect(AnthropicLanguageModel.CustomGenerationOptions.ServiceTier.standard.rawValue == "standard")
        #expect(AnthropicLanguageModel.CustomGenerationOptions.ServiceTier.priority.rawValue == "priority")
    }
}

@Suite("OpenAI CustomGenerationOptions")
struct OpenAICustomOptionsTests {
    @Test func initialization() {
        let options = OpenAILanguageModel.CustomGenerationOptions(
            topP: 0.9,
            frequencyPenalty: 0.5,
            presencePenalty: 0.3,
            stopSequences: ["END", "STOP"],
            logitBias: [123: 50, 456: -50],
            seed: 42,
            logprobs: true,
            topLogprobs: 5,
            numberOfCompletions: 2,
            verbosity: .medium,
            reasoningEffort: .high,
            reasoning: .init(effort: .medium, summary: "concise"),
            parallelToolCalls: false,
            maxToolCalls: 10,
            serviceTier: .priority,
            store: true,
            metadata: ["key": "value"],
            safetyIdentifier: "user-123",
            promptCacheKey: "cache-key",
            promptCacheRetention: "24h",
            truncation: .auto,
            extraBody: ["custom_param": .string("value")]
        )

        #expect(options.topP == 0.9)
        #expect(options.frequencyPenalty == 0.5)
        #expect(options.presencePenalty == 0.3)
        #expect(options.stopSequences == ["END", "STOP"])
        #expect(options.logitBias == [123: 50, 456: -50])
        #expect(options.seed == 42)
        #expect(options.logprobs == true)
        #expect(options.topLogprobs == 5)
        #expect(options.numberOfCompletions == 2)
        #expect(options.verbosity == .medium)
        #expect(options.reasoningEffort == .high)
        #expect(options.reasoning?.effort == .medium)
        #expect(options.reasoning?.summary == "concise")
        #expect(options.parallelToolCalls == false)
        #expect(options.maxToolCalls == 10)
        #expect(options.serviceTier == .priority)
        #expect(options.store == true)
        #expect(options.metadata == ["key": "value"])
        #expect(options.safetyIdentifier == "user-123")
        #expect(options.promptCacheKey == "cache-key")
        #expect(options.promptCacheRetention == "24h")
        #expect(options.truncation == .auto)
        #expect(options.extraBody?["custom_param"] == .string("value"))
    }

    @Test func equality() {
        let options1 = OpenAILanguageModel.CustomGenerationOptions(
            topP: 0.9,
            frequencyPenalty: 0.5,
            stopSequences: ["END"]
        )
        let options2 = OpenAILanguageModel.CustomGenerationOptions(
            topP: 0.9,
            frequencyPenalty: 0.5,
            stopSequences: ["END"]
        )

        #expect(options1 == options2)
    }

    @Test func codable() throws {
        let options = OpenAILanguageModel.CustomGenerationOptions(
            topP: 0.9,
            frequencyPenalty: 0.5,
            presencePenalty: 0.3,
            stopSequences: ["END"],
            seed: 42,
            logprobs: true,
            topLogprobs: 5,
            reasoningEffort: .high,
            serviceTier: .priority,
            store: true,
            metadata: ["key": "value"],
            truncation: .auto
        )

        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(
            OpenAILanguageModel.CustomGenerationOptions.self,
            from: data
        )

        #expect(decoded == options)
    }

    @Test func codableUsesSnakeCase() throws {
        let options = OpenAILanguageModel.CustomGenerationOptions(
            topP: 0.9,
            frequencyPenalty: 0.5,
            topLogprobs: 5,
            reasoningEffort: .medium,
            parallelToolCalls: true
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(options)
        let json = String(data: data, encoding: .utf8)!

        #expect(json.contains("\"top_p\""))
        #expect(json.contains("\"frequency_penalty\""))
        #expect(json.contains("\"top_logprobs\""))
        #expect(json.contains("\"reasoning_effort\""))
        #expect(json.contains("\"parallel_tool_calls\""))
    }

    @Test func nilProperties() {
        let options = OpenAILanguageModel.CustomGenerationOptions()
        #expect(options.topP == nil)
        #expect(options.frequencyPenalty == nil)
        #expect(options.presencePenalty == nil)
        #expect(options.stopSequences == nil)
        #expect(options.logitBias == nil)
        #expect(options.seed == nil)
        #expect(options.logprobs == nil)
        #expect(options.topLogprobs == nil)
        #expect(options.numberOfCompletions == nil)
        #expect(options.verbosity == nil)
        #expect(options.reasoningEffort == nil)
        #expect(options.reasoning == nil)
        #expect(options.parallelToolCalls == nil)
        #expect(options.maxToolCalls == nil)
        #expect(options.serviceTier == nil)
        #expect(options.store == nil)
        #expect(options.metadata == nil)
        #expect(options.safetyIdentifier == nil)
        #expect(options.promptCacheKey == nil)
        #expect(options.promptCacheRetention == nil)
        #expect(options.truncation == nil)
        #expect(options.extraBody == nil)
    }

    @Test func integrationWithGenerationOptions() {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OpenAILanguageModel.self] = .init(
            topP: 0.9,
            frequencyPenalty: 0.5,
            presencePenalty: 0.3,
            stopSequences: ["END"],
            seed: 42,
            reasoningEffort: .high,
            serviceTier: .priority
        )

        let retrieved = options[custom: OpenAILanguageModel.self]
        #expect(retrieved?.topP == 0.9)
        #expect(retrieved?.frequencyPenalty == 0.5)
        #expect(retrieved?.presencePenalty == 0.3)
        #expect(retrieved?.stopSequences == ["END"])
        #expect(retrieved?.seed == 42)
        #expect(retrieved?.reasoningEffort == .high)
        #expect(retrieved?.serviceTier == .priority)
    }

    @Test func verbosityValues() {
        #expect(OpenAILanguageModel.CustomGenerationOptions.Verbosity.low.rawValue == "low")
        #expect(OpenAILanguageModel.CustomGenerationOptions.Verbosity.medium.rawValue == "medium")
        #expect(OpenAILanguageModel.CustomGenerationOptions.Verbosity.high.rawValue == "high")
    }

    @Test func reasoningEffortValues() {
        #expect(OpenAILanguageModel.CustomGenerationOptions.ReasoningEffort.none.rawValue == "none")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ReasoningEffort.minimal.rawValue == "minimal")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ReasoningEffort.low.rawValue == "low")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ReasoningEffort.medium.rawValue == "medium")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ReasoningEffort.high.rawValue == "high")
    }

    @Test func serviceTierValues() {
        #expect(OpenAILanguageModel.CustomGenerationOptions.ServiceTier.auto.rawValue == "auto")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ServiceTier.default.rawValue == "default")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ServiceTier.flex.rawValue == "flex")
        #expect(OpenAILanguageModel.CustomGenerationOptions.ServiceTier.priority.rawValue == "priority")
    }

    @Test func truncationValues() {
        #expect(OpenAILanguageModel.CustomGenerationOptions.Truncation.auto.rawValue == "auto")
        #expect(OpenAILanguageModel.CustomGenerationOptions.Truncation.disabled.rawValue == "disabled")
    }

    @Test func reasoningConfigurationCodable() throws {
        let config = OpenAILanguageModel.CustomGenerationOptions.ReasoningConfiguration(
            effort: .high,
            summary: "detailed"
        )

        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(
            OpenAILanguageModel.CustomGenerationOptions.ReasoningConfiguration.self,
            from: data
        )

        #expect(decoded == config)
    }

    @Test func extraBodyStillWorks() {
        let options = OpenAILanguageModel.CustomGenerationOptions(
            extraBody: [
                "reasoning": .object(["enabled": .bool(true)]),
                "custom_param": .string("value"),
            ]
        )

        #expect(options.extraBody?.count == 2)
        #expect(options.extraBody?["reasoning"] == .object(["enabled": .bool(true)]))
    }
}

@Suite("OpenResponses CustomGenerationOptions")
struct OpenResponsesCustomOptionsTests {
    @Test func initialization() {
        let options = OpenResponsesLanguageModel.CustomGenerationOptions(
            toolChoice: .function(name: "getWeather"),
            allowedTools: ["getWeather", "sendEmail"],
            topP: 0.9,
            presencePenalty: 0.3,
            frequencyPenalty: 0.5,
            parallelToolCalls: false,
            maxToolCalls: 10,
            reasoningEffort: .high,
            reasoning: .init(effort: .medium, summary: "concise"),
            verbosity: .medium,
            maxOutputTokens: 100,
            store: true,
            metadata: ["key": "value"],
            safetyIdentifier: "user-123",
            truncation: .auto,
            extraBody: ["custom_param": .string("value")]
        )
        #expect(options.toolChoice != nil)
        if case .function(name: let n) = options.toolChoice! {
            #expect(n == "getWeather")
        }
        #expect(options.allowedTools?.count == 2)
        #expect(options.topP == 0.9)
        #expect(options.presencePenalty == 0.3)
        #expect(options.frequencyPenalty == 0.5)
        #expect(options.parallelToolCalls == false)
        #expect(options.maxToolCalls == 10)
        #expect(options.reasoningEffort == .high)
        #expect(options.reasoning?.effort == .medium)
        #expect(options.reasoning?.summary == "concise")
        #expect(options.verbosity == .medium)
        #expect(options.maxOutputTokens == 100)
        #expect(options.store == true)
        #expect(options.metadata?["key"] == "value")
        #expect(options.safetyIdentifier == "user-123")
        #expect(options.truncation == .auto)
        #expect(options.extraBody?["custom_param"] == .string("value"))
    }

    @Test func equality() {
        let options1 = OpenResponsesLanguageModel.CustomGenerationOptions(
            toolChoice: .auto,
            topP: 0.9
        )
        let options2 = OpenResponsesLanguageModel.CustomGenerationOptions(
            toolChoice: .auto,
            topP: 0.9
        )
        #expect(options1 == options2)
    }

    @Test func codable() throws {
        let options = OpenResponsesLanguageModel.CustomGenerationOptions(
            toolChoice: .required,
            topP: 0.9,
            truncation: .auto,
            extraBody: ["k": .string("v")]
        )
        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(
            OpenResponsesLanguageModel.CustomGenerationOptions.self,
            from: data
        )
        #expect(decoded == options)
    }

    @Test func codableUsesSnakeCase() throws {
        typealias ToolChoice = OpenResponsesLanguageModel.CustomGenerationOptions.ToolChoice
        let options = OpenResponsesLanguageModel.CustomGenerationOptions(
            toolChoice: ToolChoice.none,
            topP: 0.9,
            parallelToolCalls: true
        )
        let data = try JSONEncoder().encode(options)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("\"top_p\""))
        #expect(json.contains("\"parallel_tool_calls\""))
        #expect(json.contains("\"tool_choice\""))
    }

    @Test func toolChoiceVariants() {
        typealias ToolChoice = OpenResponsesLanguageModel.CustomGenerationOptions.ToolChoice
        let none = OpenResponsesLanguageModel.CustomGenerationOptions(toolChoice: ToolChoice.none)
        let auto = OpenResponsesLanguageModel.CustomGenerationOptions(toolChoice: .auto)
        let required = OpenResponsesLanguageModel.CustomGenerationOptions(toolChoice: .required)
        let fn = OpenResponsesLanguageModel.CustomGenerationOptions(toolChoice: .function(name: "x"))
        let allowed = OpenResponsesLanguageModel.CustomGenerationOptions(
            toolChoice: .allowedTools(tools: ["a"], mode: .auto)
        )
        #expect(none.toolChoice != auto.toolChoice)
        #expect(auto.toolChoice != required.toolChoice)
        #expect(required.toolChoice != fn.toolChoice)
        #expect(fn.toolChoice != allowed.toolChoice)
    }

    @Test func toolChoiceAllowedToolsCodable() throws {
        typealias ToolChoice = OpenResponsesLanguageModel.CustomGenerationOptions.ToolChoice

        // Round-trip: spec format (tools as {type,name} objects) with explicit mode
        let withMode: ToolChoice = .allowedTools(tools: ["get_weather", "send_email"], mode: .required)
        let encodedWithMode = try JSONEncoder().encode(withMode)
        let decodedWithMode = try JSONDecoder().decode(ToolChoice.self, from: encodedWithMode)
        #expect(decodedWithMode == withMode)

        // Decode from spec-compliant JSON (tools array of objects)
        let specJSON = """
            {"type":"allowed_tools","tools":[{"type":"function","name":"a"},{"type":"function","name":"b"}],"mode":"none"}
            """
        let fromSpec = try JSONDecoder().decode(ToolChoice.self, from: Data(specJSON.utf8))
        #expect(fromSpec == .allowedTools(tools: ["a", "b"], mode: .none))

        // Decode from string array (backwards compatibility)
        let stringsJSON = """
            {"type":"allowed_tools","tools":["x","y"]}
            """
        let fromStrings = try JSONDecoder().decode(ToolChoice.self, from: Data(stringsJSON.utf8))
        #expect(fromStrings == .allowedTools(tools: ["x", "y"], mode: .auto))

        // Encode with mode .auto: "mode" must be omitted
        let autoMode: ToolChoice = .allowedTools(tools: ["only"], mode: .auto)
        let encodedAuto = try JSONEncoder().encode(autoMode)
        let jsonAuto = String(data: encodedAuto, encoding: .utf8)!
        #expect(!jsonAuto.contains("\"mode\""))

        // Round-trip allowedTools with mode .auto
        let decodedAuto = try JSONDecoder().decode(ToolChoice.self, from: encodedAuto)
        #expect(decodedAuto == autoMode)
    }

    @Test func integrationWithGenerationOptions() {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OpenResponsesLanguageModel.self] = .init(
            toolChoice: .auto,
            allowedTools: ["getWeather"],
            topP: 0.9
        )
        let retrieved = options[custom: OpenResponsesLanguageModel.self]
        #expect(retrieved?.toolChoice == .auto)
        #expect(retrieved?.topP == 0.9)
        #expect(retrieved?.allowedTools?.count == 1)
    }

    @Test func nilProperties() {
        let options = OpenResponsesLanguageModel.CustomGenerationOptions()
        #expect(options.toolChoice == nil)
        #expect(options.allowedTools == nil)
        #expect(options.extraBody == nil)
    }
}

@Suite("Ollama CustomGenerationOptions")
struct OllamaCustomOptionsTests {
    @Test func typealiasIsDictionary() {
        // CustomGenerationOptions is a typealias to [String: JSONValue]
        let options: OllamaLanguageModel.CustomGenerationOptions = [
            "seed": .int(42),
            "repeat_penalty": .double(1.2),
        ]

        #expect(options.count == 2)
        #expect(options["seed"] == .int(42))
        #expect(options["repeat_penalty"] == .double(1.2))
    }

    @Test func equality() {
        let options1: OllamaLanguageModel.CustomGenerationOptions = [
            "seed": .int(42)
        ]
        let options2: OllamaLanguageModel.CustomGenerationOptions = [
            "seed": .int(42)
        ]

        #expect(options1 == options2)
    }

    @Test func codable() throws {
        let options: OllamaLanguageModel.CustomGenerationOptions = [
            "seed": .int(42),
            "stop": .array([.string("END")]),
        ]

        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(
            OllamaLanguageModel.CustomGenerationOptions.self,
            from: data
        )

        #expect(decoded == options)
    }

    @Test func integrationWithGenerationOptions() {
        var options = GenerationOptions(temperature: 0.8)
        options[custom: OllamaLanguageModel.self] = [
            "seed": .int(42),
            "repeat_penalty": .double(1.1),
            "stop": .array([.string("END"), .string("STOP")]),
        ]

        let retrieved = options[custom: OllamaLanguageModel.self]
        #expect(retrieved?["seed"] == .int(42))
        #expect(retrieved?["repeat_penalty"] == .double(1.1))
        #expect(retrieved?["stop"] == .array([.string("END"), .string("STOP")]))
    }

    @Test func emptyOptions() {
        let options: OllamaLanguageModel.CustomGenerationOptions = [:]
        #expect(options.isEmpty)
    }

    @Test func variousValueTypes() {
        let options: OllamaLanguageModel.CustomGenerationOptions = [
            "num_ctx": .int(4096),
            "top_p": .double(0.9),
            "penalize_newline": .bool(false),
            "stop": .array([.string("###")]),
        ]

        #expect(options["num_ctx"] == .int(4096))
        #expect(options["top_p"] == .double(0.9))
        #expect(options["penalize_newline"] == .bool(false))
        #expect(options["stop"] == .array([.string("###")]))
    }
}

@Suite("Gemini CustomGenerationOptions")
struct GeminiCustomOptionsTests {
    typealias Thinking = GeminiLanguageModel.CustomGenerationOptions.Thinking
    typealias ServerTool = GeminiLanguageModel.CustomGenerationOptions.ServerTool

    @Test func initialization() {
        let options = GeminiLanguageModel.CustomGenerationOptions(
            thinking: .dynamic,
            serverTools: [.googleSearch, .codeExecution]
        )

        #expect(options.thinking == .dynamic)
        #expect(options.serverTools?.count == 2)
    }

    @Test func thinkingModes() {
        let disabled = GeminiLanguageModel.CustomGenerationOptions(thinking: .disabled)
        let dynamic = GeminiLanguageModel.CustomGenerationOptions(thinking: .dynamic)
        let budgeted = GeminiLanguageModel.CustomGenerationOptions(thinking: .budget(1024))

        #expect(disabled.thinking == .disabled)
        #expect(dynamic.thinking == .dynamic)
        #expect(budgeted.thinking == .budget(1024))
    }

    @Test func thinkingExpressibleByLiteral() {
        let fromBool: Thinking = true
        let fromBoolFalse: Thinking = false
        let fromInt: Thinking = 2048

        #expect(fromBool == .dynamic)
        #expect(fromBoolFalse == .disabled)
        #expect(fromInt == .budget(2048))
    }

    @Test func serverToolTypes() {
        let options = GeminiLanguageModel.CustomGenerationOptions(
            serverTools: [
                .googleSearch,
                .urlContext,
                .codeExecution,
                .googleMaps(latitude: 37.7749, longitude: -122.4194),
            ]
        )

        #expect(options.serverTools?.count == 4)
    }

    @Test func equality() {
        let options1 = GeminiLanguageModel.CustomGenerationOptions(
            thinking: .dynamic,
            serverTools: [.googleSearch]
        )
        let options2 = GeminiLanguageModel.CustomGenerationOptions(
            thinking: .dynamic,
            serverTools: [.googleSearch]
        )

        #expect(options1 == options2)
    }

    @Test func inequality() {
        let options1 = GeminiLanguageModel.CustomGenerationOptions(thinking: .dynamic)
        let options2 = GeminiLanguageModel.CustomGenerationOptions(thinking: .disabled)
        let options3 = GeminiLanguageModel.CustomGenerationOptions(thinking: .budget(1024))

        #expect(options1 != options2)
        #expect(options1 != options3)
        #expect(options2 != options3)
    }

    @Test func integrationWithGenerationOptions() {
        var options = GenerationOptions(temperature: 0.7)
        options[custom: GeminiLanguageModel.self] = .init(
            thinking: .dynamic,
            serverTools: [.googleSearch, .codeExecution]
        )

        let retrieved = options[custom: GeminiLanguageModel.self]
        #expect(retrieved?.thinking == .dynamic)
        #expect(retrieved?.serverTools?.count == 2)
    }

    @Test func nilValuesUseModelDefaults() {
        let options = GeminiLanguageModel.CustomGenerationOptions()

        #expect(options.thinking == nil)
        #expect(options.serverTools == nil)
    }

    @Test func serverToolEquality() {
        let tool1 = ServerTool.googleMaps(latitude: 37.7749, longitude: -122.4194)
        let tool2 = ServerTool.googleMaps(latitude: 37.7749, longitude: -122.4194)
        let tool3 = ServerTool.googleMaps(latitude: 40.7128, longitude: -74.0060)

        #expect(tool1 == tool2)
        #expect(tool1 != tool3)
        #expect(ServerTool.googleSearch == ServerTool.googleSearch)
    }
}

#if Llama
    @Suite("Llama CustomGenerationOptions")
    struct LlamaCustomOptionsTests {
        @Test func initialization() {
            let options = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.2,
                repeatLastN: 128,
                frequencyPenalty: 0.1,
                presencePenalty: 0.1
            )

            #expect(options.repeatPenalty == 1.2)
            #expect(options.repeatLastN == 128)
            #expect(options.frequencyPenalty == 0.1)
            #expect(options.presencePenalty == 0.1)
        }

        @Test func mirostatModes() {
            let v1 = LlamaLanguageModel.CustomGenerationOptions(
                mirostat: .v1(tau: 5.0, eta: 0.1)
            )
            let v2 = LlamaLanguageModel.CustomGenerationOptions(
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            #expect(v1.mirostat != nil)
            #expect(v2.mirostat != nil)
            #expect(v1 != v2)
        }

        @Test func equality() {
            let options1 = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.1,
                repeatLastN: 64
            )
            let options2 = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.1,
                repeatLastN: 64
            )

            #expect(options1 == options2)
        }

        @Test func codable() throws {
            let options = LlamaLanguageModel.CustomGenerationOptions(
                repeatPenalty: 1.2,
                repeatLastN: 128,
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            let data = try JSONEncoder().encode(options)
            let decoded = try JSONDecoder().decode(
                LlamaLanguageModel.CustomGenerationOptions.self,
                from: data
            )

            #expect(decoded == options)
        }

        @Test func integrationWithGenerationOptions() {
            var options = GenerationOptions(temperature: 0.8)
            options[custom: LlamaLanguageModel.self] = .init(
                repeatPenalty: 1.2,
                repeatLastN: 128
            )

            let retrieved = options[custom: LlamaLanguageModel.self]
            #expect(retrieved?.repeatPenalty == 1.2)
            #expect(retrieved?.repeatLastN == 128)
        }
    }
#endif
