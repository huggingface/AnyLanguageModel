import EventSource
import Foundation
import JSONSchema
import OrderedCollections

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A language model that connects to Anthropic's Claude API.
///
/// Use this model to generate text using Claude models from Anthropic.
///
/// ```swift
/// let model = AnthropicLanguageModel(
///     apiKey: "your-api-key",
///     model: "claude-3-5-sonnet-20241022"
/// )
/// ```
///
/// You can also specify beta headers to access experimental features:
///
/// ```swift
/// let model = AnthropicLanguageModel(
///     apiKey: "your-api-key",
///     model: "claude-3-5-sonnet-20241022",
///     betas: ["beta1", "beta2"]
/// )
/// ```
public struct AnthropicLanguageModel: LanguageModel {
    /// Custom generation options specific to Anthropic's Claude API.
    ///
    /// Use this type to pass additional parameters that are not part of the
    /// standard ``GenerationOptions``, such as Anthropic-specific sampling
    /// parameters and metadata.
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: AnthropicLanguageModel.self] = .init(
    ///     topP: 0.9,
    ///     topK: 40,
    ///     stopSequences: ["END", "STOP"]
    /// )
    /// ```
    public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions, Codable {
        /// Use nucleus sampling with probability mass `topP`.
        ///
        /// In nucleus sampling, tokens are sorted by probability and added to a
        /// pool until the cumulative probability exceeds `topP`. A token is then
        /// sampled from the pool. We recommend altering either `temperature` or
        /// `topP`, but not both.
        ///
        /// Recommended range: `0.0` to `1.0`. Defaults to `nil` (not specified).
        public var topP: Double?

        /// Only sample from the top K options for each subsequent token.
        ///
        /// Used to remove "long tail" low probability responses. We recommend
        /// using `topP` instead, or combining `topK` with `topP`.
        ///
        /// Recommended range: `0` to `500`. Defaults to `nil` (not specified).
        public var topK: Int?

        /// Custom text sequences that will cause the model to stop generating.
        ///
        /// Our models will normally stop when they have naturally completed their turn,
        /// which will result in a response `stop_reason` of `"end_turn"`.
        ///
        /// If you want the model to stop generating when it encounters custom strings
        /// of text, you can use the `stop_sequences` parameter. If the model encounters
        /// one of the custom sequences, the response `stop_reason` value will be
        /// `"stop_sequence"` and the response `stop_sequence` value will contain the
        /// matched stop sequence.
        public var stopSequences: [String]?

        /// An object describing metadata about the request.
        public var metadata: Metadata?

        /// How the model should use the provided tools.
        ///
        /// Use this to control whether the model can use tools and which tools it prefers.
        public var toolChoice: ToolChoice?

        /// Configuration for extended thinking.
        ///
        /// When enabled, the model will use internal reasoning before responding,
        /// which can improve performance on complex tasks.
        public var thinking: Thinking?

        /// Specifies the tier of service to use for the request.
        ///
        /// The default is "auto", which will use the priority tier if available
        /// and fall back to standard.
        public var serviceTier: ServiceTier?

        /// Additional parameters to include in the request body.
        ///
        /// These parameters are merged into the top-level request JSON,
        /// allowing you to pass additional options not explicitly modeled.
        public var extraBody: [String: JSONValue]?
        
        public var effort: Effort?

        // MARK: - Nested Types

        /// Metadata about the request.
        public struct Metadata: Hashable, Codable, Sendable {
            /// An external identifier for the user who is associated with the request.
            ///
            /// This should be a UUID, hash value, or other opaque identifier.
            /// Anthropic may use this ID to help detect abuse. Do not include any
            /// identifying information such as name, email address, or phone number.
            public var userID: String?

            enum CodingKeys: String, CodingKey {
                case userID = "user_id"
            }

            /// Creates metadata for an Anthropic request.
            ///
            /// - Parameter userID: An external identifier for the user.
            public init(userID: String? = nil) {
                self.userID = userID
            }
        }

        /// Controls how the model uses tools.
        public enum ToolChoice: Hashable, Codable, Sendable {
            /// The model automatically decides whether to use tools.
            case auto

            /// The model must use one of the provided tools.
            case any

            /// The model must use the specified tool.
            case tool(name: String)

            /// The model will not be allowed to use tools.
            case disabled

            enum CodingKeys: String, CodingKey {
                case type
                case name
                case disableParallelToolUse = "disable_parallel_tool_use"
            }

            public init(from decoder: any Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                let type = try container.decode(String.self, forKey: .type)

                switch type {
                case "auto":
                    self = .auto
                case "any":
                    self = .any
                case "tool":
                    let name = try container.decode(String.self, forKey: .name)
                    self = .tool(name: name)
                case "none":
                    self = .disabled
                default:
                    self = .auto
                }
            }

            public func encode(to encoder: any Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)

                switch self {
                case .auto:
                    try container.encode("auto", forKey: .type)
                case .any:
                    try container.encode("any", forKey: .type)
                case .tool(let name):
                    try container.encode("tool", forKey: .type)
                    try container.encode(name, forKey: .name)
                case .disabled:
                    try container.encode("none", forKey: .type)
                }
            }
        }

        /// How much effort the model should put into a task.
        ///
        /// Docs: https://platform.claude.com/docs/en/build-with-claude/effort
        public enum Effort: String, Hashable, Codable, Sendable {
            /// Absolute maximum capability with no constraints on token spending.
            ///
            /// Use Case: Tasks requiring the deepest possible reasoning and most thorough analysis
            /// Availability: Claude Fable 5, Claude Mythos 5, Claude Opus 4.8, Claude Mythos Preview, Claude Opus 4.7, Claude Opus 4.6, Claude Sonnet 5, and Claude Sonnet 4.6.
            case max
            /// Extended capability for long-horizon work.
            /// Use Case: Long-running agentic and coding tasks (over 30 minutes) with token budgets in the millions
            /// Availability: Claude Fable 5, Claude Mythos 5, Claude Opus 4.8, Claude Opus 4.7, and Claude Sonnet 5.
            case extraHigh = "xHigh"
            /// High capability. Equivalent to not setting the parameter.
            /// Use Case: Complex reasoning, difficult coding problems, agentic tasks
            /// Availability: All Models
            case high
            /// Balanced approach with moderate token savings.
            /// Use Case: Agentic tasks that require a balance of speed, cost, and performance
            /// Availability: All Models
            case medium
            /// Most efficient. Significant token savings with some capability reduction.
            /// Use Case: Simpler tasks that need the best speed and lowest costs, like subagents
            /// Availability: All Models
            case low
        }
        
        /// Configuration for extended thinking.
        public struct Thinking: Hashable, Codable, Sendable {
            /// The type of thinking to use.
            public var type: ThinkingType

            /// The maximum number of tokens to use for thinking. Nil when `type` = `.adaptive`.
            ///
            /// This budget is the maximum number of tokens the model can use for its
            /// internal reasoning process. Larger budgets can improve response quality
            /// for complex tasks but increase latency and cost.
            public var budgetTokens: Int?
            
            /// How thinking should be displayed.
            public var display: ThinkingDisplay?

            /// The type of thinking mode.
            public enum ThinkingType: String, Hashable, Codable, Sendable {
                /// Enables extended thinking.
                case enabled
                /// Enables adaptive thinking.
                case adaptive
            }
            
            /// How thinking should be returned during generation.
            public enum ThinkingDisplay: String, Hashable, Codable, Sendable {
                /// Thinking will be summarized.
                case summarized
                /// No thoughts will be returned.
                case omitted
            }

            enum CodingKeys: String, CodingKey {
                case type
                case budgetTokens = "budget_tokens"
                case display
            }

            /// Creates a thinking configuration.
            /// 
            /// - Parameters:
            ///   - type: The type of thinking to perform.
            ///   - budgetTokens: The maximum number of tokens to use for thinking. Only required when `type` == `.enabled`.
            ///   - display: The display type for thoughts.
            public init(type: ThinkingType, budgetTokens: Int?, display: ThinkingDisplay?) {
                self.type = type
                self.budgetTokens = budgetTokens
                self.display = display
            }
            
            /// Convenience function for enabling adaptive thinking on supported models.
            public static func adaptive(display: ThinkingDisplay?) -> Thinking {
                return Thinking.init(type: .adaptive, budgetTokens: nil, display: display)
            }
            
            /// Convenience function for enabling thinking with a token budget on supported models.
            public static func enabled(budgetTokens: Int, display: ThinkingDisplay?) -> Thinking {
                return Thinking.init(type: .enabled, budgetTokens: budgetTokens, display: display)
            }
        }

        /// The tier of service for processing the request.
        public enum ServiceTier: String, Hashable, Codable, Sendable {
            /// Automatically select the best available tier.
            case auto

            /// Standard tier processing.
            case standard

            /// Priority tier processing with faster response times.
            case priority
        }

        /// Creates custom generation options for Anthropic's Claude API.
        ///
        /// - Parameters:
        ///   - topP: Use nucleus sampling with this probability mass.
        ///   - topK: Only sample from the top K options for each token.
        ///   - stopSequences: Custom text sequences that will cause the model to stop generating.
        ///   - metadata: An object describing metadata about the request.
        ///   - toolChoice: How the model should use the provided tools.
        ///   - thinking: Configuration for extended thinking.
        ///   - serviceTier: The tier of service to use for the request.
        ///   - extraBody: Additional parameters to include in the request body.
        public init(
            topP: Double? = nil,
            topK: Int? = nil,
            stopSequences: [String]? = nil,
            metadata: Metadata? = nil,
            toolChoice: ToolChoice? = nil,
            thinking: Thinking? = nil,
            serviceTier: ServiceTier? = nil,
            extraBody: [String: JSONValue]? = nil,
            effort: Effort? = nil
        ) {
            self.topP = topP
            self.topK = topK
            self.stopSequences = stopSequences
            self.metadata = metadata
            self.toolChoice = toolChoice
            self.thinking = thinking
            self.serviceTier = serviceTier
            self.extraBody = extraBody
            self.effort = effort
        }
    }
    /// The reason the model is unavailable.
    /// This model is always available.
    public typealias UnavailableReason = Never

    /// The default base URL for Anthropic's API.
    public static let defaultBaseURL = URL(string: "https://api.anthropic.com")!

    /// The default API version for Anthropic's API.
    public static let defaultAPIVersion = "2023-06-01"

    /// The base URL for the API endpoint.
    public let baseURL: URL

    /// The closure providing the API key for authentication.
    private let tokenProvider: @Sendable () -> String

    /// The API version to use for requests.
    public let apiVersion: String

    /// Optional beta version(s) of the API to use.
    public let betas: [String]?

    /// The model identifier to use for generation.
    public let model: String

    private let httpSession: HTTPSession

    /// Creates an Anthropic language model.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the API endpoint. Defaults to Anthropic's official API.
    ///   - apiKey: Your Anthropic API key or a closure that returns it.
    ///   - apiVersion: The API version to use for requests. Defaults to `2023-06-01`.
    ///   - betas: Optional beta version(s) of the API to use.
    ///   - model: The model identifier (for example, "claude-3-5-sonnet-20241022").
    ///   - session: The HTTP session or client used for network requests.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        apiVersion: String = defaultAPIVersion,
        betas: [String]? = nil,
        model: String,
        session: HTTPSession = makeDefaultSession(),
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.apiVersion = apiVersion
        self.betas = betas
        self.model = model
        self.httpSession = session
    }

    public func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let url = baseURL.appendingPathComponent("v1/messages")
        let headers = buildHeaders()

        // Convert available tools to Anthropic format
        let anthropicTools: [AnthropicTool] = try session.tools.map { tool in
            try convertToolToAnthropicFormat(tool)
        }
        
        let responseSchema = type == String.self ? nil : try convertSchemaToAnthropicFormat(Content.generationSchema)
        
        var entries: [Transcript.Entry] = []
        var runningText = ""
        var messages: [AnthropicMessage] = session.transcript.toAnthropicMessages()
        
        // Loop until no more tool calls are found.
        while true {
            let params = try createMessageParams(
                model: model,
                system: nil,
                messages: messages,
                tools: anthropicTools.isEmpty ? nil : anthropicTools,
                responseSchema: responseSchema,
                options: options
            )
            
            let body = try JSONEncoder().encode(params)

            let message: AnthropicMessageResponse = try await httpSession.fetch(
                .post,
                url: url,
                headers: headers,
                body: body
            )

            // Append to messages for future response loops.
            messages.append(AnthropicMessage(role: .assistant, content: message.content))
            
            // Handle tool calls, if present
            let toolUses: [AnthropicToolUse] = message.content.compactMap { content in
                if case .toolUse(let u) = content { return u }
                return nil
            }

            if !toolUses.isEmpty {
                let resolution = try await resolveToolUses(toolUses, session: session)
                switch resolution {
                case .stop(let calls):
                    if !calls.isEmpty {
                        entries.append(.toolCalls(Transcript.ToolCalls(calls)))
                    }
                    let empty = try emptyResponseContent(for: type)
                    return LanguageModelSession.Response(
                        content: empty.content,
                        rawContent: empty.rawContent,
                        transcriptEntries: ArraySlice(entries)
                    )
                case .invocations(let invocations):
                    if !invocations.isEmpty {
                        var toolResultBlocks: [AnthropicContent] = []

                        for invocation in invocations {
                            entries.append(.toolOutput(invocation.output))
                            toolResultBlocks.append(
                                .toolResult(
                                    AnthropicToolResult(
                                        toolUseId: invocation.call.id,
                                        content: convertSegmentsToAnthropicContent(invocation.output.segments)
                                    )
                                )
                            )
                        }
                        
                        messages.append(AnthropicMessage(role: .user, content: toolResultBlocks))
                        entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                        
                        continue // Keep going through the loop
                    }
                }
            }

            // If we make it here, no tools were called and we can assume this is the last message.
            runningText = message.content.compactMap { block -> String? in
                switch block {
                case .text(let t): return t.text
                default: return nil
                }
            }.joined()
            
            break // Break the loop
        }

        if type == String.self {
            return LanguageModelSession.Response(
                content: runningText as! Content,
                rawContent: GeneratedContent(runningText),
                transcriptEntries: ArraySlice(entries)
            )
        }

        let rawContent = try GeneratedContent(json: runningText)
        let content = try Content(rawContent)
        return LanguageModelSession.Response(
            content: content,
            rawContent: rawContent,
            transcriptEntries: ArraySlice(entries)
        )
    }
    
    
    struct ContentAccumulationBlocks: Hashable, Codable, Sendable {
        enum Kind: Hashable, Codable, Sendable { case toolUse, thinking, text }
        
        var kind: Kind
        var text: String
        var partialJSON: String?

        // Used by tool use content blocks
        var id: String?
        var name: String?
        var signature: String?
    }

    public func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let url = baseURL.appendingPathComponent("v1/messages")

        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
            continuation in
            let task = Task { @Sendable in
                do {
                    let headers = buildHeaders()

                    // Convert available tools to Anthropic format
                    let anthropicTools: [AnthropicTool] = try session.tools.map { tool in
                        try convertToolToAnthropicFormat(tool)
                    }

                    let responseSchema = type == String.self ? nil : try convertSchemaToAnthropicFormat(Content.generationSchema)
                    let expectsStructuredResponse = type != String.self
                    
                    var messages: [AnthropicMessage] = session.transcript.toAnthropicMessages()

                    while true {
                        var accumulatedText: String = ""
                        var stopReason: String? = nil

                        let params = try createMessageParams(
                            model: model,
                            system: nil,
                            messages: messages,
                            tools: anthropicTools.isEmpty ? nil : anthropicTools,
                            responseSchema: responseSchema,
                            options: options,
                            stream: true
                        )
                        
                        let body = try JSONEncoder().encode(params)
                        
                        // Stream server-sent events from Anthropic API
                        let events: AsyncThrowingStream<AnthropicStreamEvent, any Error> =
                        httpSession
                            .fetchEventStream(
                                .post,
                                url: url,
                                headers: headers,
                                body: body
                            )
                        
                        // Accumulating content blocks keyed by their index.
                        var contentBlocks: [Int: ContentAccumulationBlocks] = [:]
                        
                        eventStream: for try await event in events {
                            switch event {
                            case .contentBlockStart(let start):
                                // Create blocks at their index
                                switch start.contentBlock.type {
                                case "tool_use":
                                    contentBlocks[start.index] = ContentAccumulationBlocks(
                                        kind: .toolUse,
                                        text: start.contentBlock.text ?? "",
                                        id: start.contentBlock.id,
                                        name: start.contentBlock.name
                                        )
                                case "thinking":
                                    contentBlocks[start.index] = ContentAccumulationBlocks(
                                        kind: .thinking,
                                        text: start.contentBlock.text ?? ""
                                    )
                                default:
                                    contentBlocks[start.index] = ContentAccumulationBlocks(
                                        kind: .text,
                                        text: start.contentBlock.text ?? ""
                                    )
                                }
                            case .contentBlockDelta(let delta):
                                switch delta.delta {
                                case .textDelta(let textDelta):
                                    // Accumulate text delta for streaming
                                    // Make sure the block has even been started.
                                    guard  contentBlocks[delta.index] != nil else { continue }
                                     
                                    // Set default text
                                    if contentBlocks[delta.index]?.text == nil {
                                        contentBlocks[delta.index]?.text = ""
                                    }
                                    
                                    contentBlocks[delta.index]?.text += textDelta.text
                                    accumulatedText += textDelta.text
                                    
                                    // Grow the observable transcript so a Transcript-driven UI updates live.
                                    session.growStreamingTranscript(text: accumulatedText)
                                    
                                    // Send text back normally
                                    if expectsStructuredResponse {
                                        if let snapshot: LanguageModelSession.ResponseStream<Content>.Snapshot =
                                            try? partialSnapshot(from: accumulatedText)
                                        {
                                            continuation.yield(snapshot)
                                        }
                                    } else {
                                        let raw = GeneratedContent(accumulatedText)
                                        let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                            .asPartiallyGenerated()
                                        continuation.yield(.init(content: content, rawContent: raw))
                                    }
                                case .inputJsonDelta(let jsonDelta):
                                    if contentBlocks[delta.index]?.partialJSON == nil {
                                        contentBlocks[delta.index]?.partialJSON = ""
                                    }
                                    
                                    contentBlocks[delta.index]?.partialJSON? += jsonDelta.partialJson
                                case .thinkingDelta(let thinkingDelta):
                                    contentBlocks[delta.index]?.text += thinkingDelta.thinking
                                case .signatureDelta(let signatureDelta):
                                    if contentBlocks[delta.index]?.signature == nil {
                                        contentBlocks[delta.index]?.signature = ""
                                    }

                                    contentBlocks[delta.index]?.signature? += signatureDelta.signature
                                case .ignored:
                                    break
                                }
                            case .messageDelta(let messageDelta):
                                stopReason = messageDelta.delta.stopReason
                            case .messageStop:
                                // Need to use a label, otherwise this would break out of the switch.
                                break eventStream
                            case .ping, .ignored, .messageStart, .contentBlockStop:
                                continue
                            }
                        }
                        
                        // Assemble assistant content from the streamed content blocks
                        var assistantContent: [AnthropicContent] = []
                        var toolUses: [AnthropicToolUse] = []
                        
                        for block in contentBlocks.sorted(by: { $0.key < $1.key }).map(\.value) {
                            switch block.kind {
                            case .text:
                                assistantContent.append(
                                    AnthropicContent.text(AnthropicText(text: block.text))
                                )
                            case .thinking:
                                // Ensure there is a signature. Needed for claude to reconstruct the thought on the server.
                                guard let signature = block.signature else { continue }
                                
                                assistantContent.append(
                                    AnthropicContent.thinking(AnthropicThinking(thinking: block.text, signature: signature))
                                )
                            case .toolUse:
                                guard let id = block.id, let name = block.name, let jsonString = block.partialJSON else { continue }
                                guard let json = fromPartialJSON(jsonString) else { continue }
                                
                                let toolUse = AnthropicToolUse(id: id, name: name, input: json)
                                assistantContent.append(AnthropicContent.toolUse(toolUse))
                                toolUses.append(toolUse)
                            }
                        }
                                
                        messages.append(AnthropicMessage(role: .assistant, content: assistantContent))
                        
                        // Process the tool calls
                        var appendedToolResults = false
                        if !toolUses.isEmpty {
                            let resolution = try await resolveToolUses(toolUses, session: session)
                            switch resolution {
                            case .stop(let calls):
                                if !calls.isEmpty {
                                    session.appendTranscriptEntry(.toolCalls(Transcript.ToolCalls(calls)))
                                }
                                continuation.finish()
                                return
                            case .invocations(let invocations):
                                if !invocations.isEmpty {
                                    var toolResultBlocks: [AnthropicContent] = []
                                    
                                    // Need to append tool calls before tool results
                                    session.appendTranscriptEntry(
                                        .toolCalls(Transcript.ToolCalls(invocations.map(\.call)))
                                    )
                                    
                                    for invocation in invocations {
                                        // Save the tool outputs into the transcript.
                                        session.appendTranscriptEntry(.toolOutput(invocation.output))
                                        
                                        toolResultBlocks.append(
                                            .toolResult(
                                                AnthropicToolResult(
                                                    toolUseId: invocation.call.id,
                                                    content: convertSegmentsToAnthropicContent(invocation.output.segments)
                                                )
                                            )
                                        )
                                    }
                                    
                                    messages.append(AnthropicMessage(role: .user, content: toolResultBlocks))
                                    appendedToolResults = true
                                }
                            }
                        }
                        
                        // Continue only if we responded with tool call results, if we didn't the turn is complete.
                        if stopReason == "tool_use" && appendedToolResults {
                            continue
                        } else {
                            break
                        }
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }

        return LanguageModelSession.ResponseStream(stream: stream)
    }

    private func buildHeaders() -> [String: String] {
        var headers: [String: String] = [
            "x-api-key": tokenProvider(),
            "anthropic-version": apiVersion,
        ]

        if let betas = betas, !betas.isEmpty {
            headers["anthropic-beta"] = betas.joined(separator: ",")
        }

        return headers
    }
}

// MARK: - Conversions

private func createMessageParams(
    model: String,
    system: String?,
    messages: [AnthropicMessage],
    tools: [AnthropicTool]?,
    responseSchema: JSONSchema?,
    options: GenerationOptions,
    stream: Bool? = nil
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "model": .string(model),
        "messages": try JSONValue(messages),
        "max_tokens": .int(options.maximumResponseTokens ?? 1024),
    ]

    if let system {
        params["system"] = .string(system)
    }
    if let tools, !tools.isEmpty {
        params["tools"] = try JSONValue(tools)
    }
    if let responseSchema {
        // Structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
        let schemaValue = try JSONValue(responseSchema)
        if case .object(let schemaObject) = schemaValue, schemaObject.isEmpty {
            // Anthropic rejects empty schemas; omit output_config in this case.
        } else {
            params["output_config"] = .object(
                [
                    "format": .object(
                        [
                            "type": .string("json_schema"),
                            "schema": schemaValue,
                        ]
                    )
                ]
            )
        }
    }
    if let temperature = options.temperature {
        params["temperature"] = .double(temperature)
    }

    // Apply Anthropic-specific custom options
    if let customOptions = options[custom: AnthropicLanguageModel.self] {
        if let topP = customOptions.topP {
            params["top_p"] = .double(topP)
        }
        if let topK = customOptions.topK {
            params["top_k"] = .int(topK)
        }
        if let stopSequences = customOptions.stopSequences, !stopSequences.isEmpty {
            params["stop_sequences"] = .array(stopSequences.map { .string($0) })
        }
        if let metadata = customOptions.metadata {
            var metadataObject: [String: JSONValue] = [:]
            if let userID = metadata.userID {
                metadataObject["user_id"] = .string(userID)
            }
            if !metadataObject.isEmpty {
                params["metadata"] = .object(metadataObject)
            }
        }
        if let toolChoice = customOptions.toolChoice {
            switch toolChoice {
            case .auto:
                params["tool_choice"] = .object(["type": .string("auto")])
            case .any:
                params["tool_choice"] = .object(["type": .string("any")])
            case .tool(let name):
                params["tool_choice"] = .object([
                    "type": .string("tool"),
                    "name": .string(name),
                ])
            case .disabled:
                params["tool_choice"] = .object(["type": .string("none")])
            }
        }
        if let effort = customOptions.effort {
            // If output_config was previously set during the response schema options, we need to append insert into that dictionary instead of replacing it.
            if let output_config = params["output_config"], var object = output_config.objectValue {
                object["effort"] = .string(effort.rawValue)
                params["output_config"] = .object(object)
            } else {
                params["output_config"] = .object(
                    [
                        "effort": .string(effort.rawValue)
                    ]
                )
            }
        }
        if let thinking = customOptions.thinking {
            var thinkingObject: [String: JSONValue] = [
                "type": .string(thinking.type.rawValue)
            ]
            if let budget = thinking.budgetTokens {
                thinkingObject["budget_tokens"] = .int(budget)
            }
            if let display = thinking.display {
                thinkingObject["display"] = .string(display.rawValue)
            }
            
            params["thinking"] = .object(thinkingObject)
        }
        if let serviceTier = customOptions.serviceTier {
            params["service_tier"] = .string(serviceTier.rawValue)
        }

        // Merge custom extraBody into the request
        if let extraBody = customOptions.extraBody {
            for (key, value) in extraBody {
                params[key] = value
            }
        }
    }

    if let stream {
        params["stream"] = .bool(stream)
    }

    return params
}

// MARK: - Tool Invocation Handling

private struct ToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private enum ToolResolutionOutcome {
    case stop(calls: [Transcript.ToolCall])
    case invocations([ToolInvocationResult])
}

private func emptyResponseContent<Content: Generable>(
    for type: Content.Type
) throws -> (content: Content, rawContent: GeneratedContent) {
    if type == String.self {
        let raw = GeneratedContent("")
        return ("" as! Content, raw)
    }

    let emptyObject = GeneratedContent(properties: [:])
    if let content = try? Content(emptyObject) {
        return (content, emptyObject)
    }

    let nullContent = GeneratedContent(kind: .null)
    if let content = try? Content(nullContent) {
        return (content, nullContent)
    }

    throw GeneratedContentConversionError.typeMismatch
}

private func partialSnapshot<Content: Generable>(
    from accumulatedText: String
) throws -> LanguageModelSession.ResponseStream<Content>.Snapshot {
    let raw = try GeneratedContent(json: accumulatedText)
    let content = try Content.PartiallyGenerated(raw)
    return .init(content: content, rawContent: raw)
}

private func convertSchemaToAnthropicFormat(_ schema: GenerationSchema) throws -> JSONSchema {
    let resolvedSchema = schema.withResolvedRoot() ?? schema
    let data = try JSONEncoder().encode(resolvedSchema)
    return try JSONDecoder().decode(JSONSchema.self, from: data)
}

private func resolveToolUses(
    _ toolUses: [AnthropicToolUse],
    session: LanguageModelSession
) async throws -> ToolResolutionOutcome {
    if toolUses.isEmpty { return .invocations([]) }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var transcriptCalls: [Transcript.ToolCall] = []
    transcriptCalls.reserveCapacity(toolUses.count)
    for use in toolUses {
        let args = try toGeneratedContent(use.input)
        let callID = use.id
        transcriptCalls.append(
            Transcript.ToolCall(
                id: callID,
                toolName: use.name,
                arguments: args
            )
        )
    }

    if let delegate = session.toolExecutionDelegate {
        await delegate.didGenerateToolCalls(transcriptCalls, in: session)
    }

    guard !transcriptCalls.isEmpty else { return .invocations([]) }

    var decisions: [ToolExecutionDecision] = []
    decisions.reserveCapacity(transcriptCalls.count)

    if let delegate = session.toolExecutionDelegate {
        for call in transcriptCalls {
            let decision = await delegate.toolCallDecision(for: call, in: session)
            if case .stop = decision {
                return .stop(calls: transcriptCalls)
            }
            decisions.append(decision)
        }
    } else {
        decisions = Array(repeating: .execute, count: transcriptCalls.count)
    }

    var results: [ToolInvocationResult] = []
    results.reserveCapacity(transcriptCalls.count)

    for (index, call) in transcriptCalls.enumerated() {
        switch decisions[index] {
        case .stop:
            // This branch should be unreachable because `.stop` returns during decision collection.
            // Keep it as a defensive guard in case that logic changes.
            return .stop(calls: transcriptCalls)
        case .provideOutput(let segments):
            let output = Transcript.ToolOutput(
                id: call.id,
                toolName: call.toolName,
                segments: segments
            )
            if let delegate = session.toolExecutionDelegate {
                await delegate.didExecuteToolCall(call, output: output, in: session)
            }
            results.append(ToolInvocationResult(call: call, output: output))
        case .execute:
            guard let tool = toolsByName[call.toolName] else {
                let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.toolName)"))
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: [message]
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(ToolInvocationResult(call: call, output: output))
                continue
            }

            do {
                let segments = try await tool.makeOutputSegments(from: call.arguments)
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: tool.name,
                    segments: segments
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(ToolInvocationResult(call: call, output: output))
            } catch {
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didFailToolCall(call, error: error, in: session)
                }
                throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
            }
        }
    }

    return .invocations(results)
}

// Convert our GenerationSchema into Anthropic's expected JSON Schema payload
private func convertToolToAnthropicFormat(_ tool: any Tool) throws -> AnthropicTool {
    let schema = try convertSchemaToAnthropicFormat(tool.parameters)
    return AnthropicTool(name: tool.name, description: tool.description, inputSchema: schema)
}

private func toGeneratedContent(_ value: [String: JSONValue]?) throws -> GeneratedContent {
    guard let value else { return GeneratedContent(properties: [:]) }
    let data = try JSONEncoder().encode(JSONValue.object(value))
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}

private func fromGeneratedContent(_ content: GeneratedContent) throws -> [String: JSONValue] {
    let data = try JSONEncoder().encode(content)
    let jsonValue = try JSONDecoder().decode(JSONValue.self, from: data)

    guard case .object(let dict) = jsonValue else {
        return [:]
    }
    return dict
}

private func fromPartialJSON(_ json: String) -> [String: JSONValue]? {
    let content = json.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !content.isEmpty else { return [:] }
    guard let data = content.data(using: .utf8) else { return nil }
    guard let jsonValue = try? JSONDecoder().decode(JSONValue.self, from: data) else { return nil }

    guard case .object(let dict) = jsonValue else {
        return nil
    }
    return dict
}


// MARK: - Supporting Types

extension Transcript {
    fileprivate func toAnthropicMessages() -> [AnthropicMessage] {
        var messages = [AnthropicMessage]()
        for item in self {
            switch item {
            case .instructions(let instructions):
                messages.append(
                    .init(
                        role: .user,
                        content: convertSegmentsToAnthropicContent(instructions.segments)
                    )
                )
            case .prompt(let prompt):
                messages.append(
                    .init(
                        role: .user,
                        content: convertSegmentsToAnthropicContent(prompt.segments)
                    )
                )
            case .response(let response):
                messages.append(
                    .init(
                        role: .assistant,
                        content: convertSegmentsToAnthropicContent(response.segments)
                    )
                )
            case .toolCalls(let toolCalls):
                // Add assistant message with tool use blocks
                let toolUseBlocks: [AnthropicContent] = toolCalls.map { call in
                    let input = try? fromGeneratedContent(call.arguments)
                    return .toolUse(
                        AnthropicToolUse(
                            id: call.id,
                            name: call.toolName,
                            input: input
                        )
                    )
                }
                
                print("Tool use block \(toolUseBlocks)")
                messages.append(
                    .init(
                        role: .assistant,
                        content: toolUseBlocks
                    )
                )
            case .toolOutput(let toolOutput):
                // Add user message with tool result
                print("tool output \(toolOutput.id)")
                messages.append(
                    .init(
                        role: .user,
                        content: [
                            .toolResult(
                                AnthropicToolResult(
                                    toolUseId: toolOutput.id,
                                    content: convertSegmentsToAnthropicContent(toolOutput.segments)
                                )
                            )
                        ]
                    )
                )
            }
        }
        return messages
    }
}

private struct AnthropicTool: Codable, Sendable {
    let name: String
    let description: String
    let inputSchema: JSONSchema

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case inputSchema = "input_schema"
    }
}

private struct AnthropicMessage: Codable, Sendable {
    enum Role: String, Codable, Sendable { case user, assistant }

    let role: Role
    let content: [AnthropicContent]
}

private enum AnthropicContent: Codable, Sendable {
    case text(AnthropicText)
    case image(AnthropicImage)
    case toolUse(AnthropicToolUse)
    case toolResult(AnthropicToolResult)
    case thinking(AnthropicThinking)

    enum CodingKeys: String, CodingKey { case type }

    enum ContentType: String, Codable {
        case text = "text", image = "image", toolUse = "tool_use", toolResult = "tool_result", thinking = "thinking"
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(ContentType.self, forKey: .type)
        switch type {
        case .text:
            self = .text(try AnthropicText(from: decoder))
        case .image:
            self = .image(try AnthropicImage(from: decoder))
        case .toolUse:
            self = .toolUse(try AnthropicToolUse(from: decoder))
        case .toolResult:
            self = .toolResult(try AnthropicToolResult(from: decoder))
        case .thinking:
            self = .thinking(try AnthropicThinking(from: decoder))
        }
    }

    func encode(to encoder: any Encoder) throws {
        switch self {
        case .text(let t): try t.encode(to: encoder)
        case .image(let i): try i.encode(to: encoder)
        case .toolUse(let u): try u.encode(to: encoder)
        case .toolResult(let r): try r.encode(to: encoder)
        case .thinking(let h): try h.encode(to: encoder)
        }
    }
}

private struct AnthropicThinking: Codable, Sendable {
    let type: String
    let thinking: String
    let signature: String

    init(thinking: String, signature: String) {
        self.type = "thinking"
        self.thinking = thinking
        self.signature = signature
    }
}

private struct AnthropicText: Codable, Sendable {
    let type: String
    let text: String

    init(text: String) {
        self.type = "text"
        self.text = text
    }
}

private struct AnthropicImage: Codable, Sendable {
    struct Source: Codable, Sendable {
        let type: String
        let mediaType: String?
        let data: String?
        let url: String?

        enum CodingKeys: String, CodingKey {
            case type
            case mediaType = "media_type"
            case data
            case url
        }
    }

    let type: String
    let source: Source

    init(base64Data: String, mimeType: String) {
        self.type = "image"
        self.source = Source(type: "base64", mediaType: mimeType, data: base64Data, url: nil)
    }

    init(url: String) {
        self.type = "image"
        self.source = Source(type: "url", mediaType: nil, data: nil, url: url)
    }
}

private func convertSegmentsToAnthropicContent(_ segments: [Transcript.Segment]) -> [AnthropicContent] {
    var blocks: [AnthropicContent] = []
    blocks.reserveCapacity(segments.count)
    for segment in segments {
        switch segment {
        case .text(let t):
            blocks.append(.text(AnthropicText(text: t.content)))
        case .structure(let s):
            blocks.append(.text(AnthropicText(text: s.content.jsonString)))
        case .image(let img):
            switch img.source {
            case .url(let url):
                blocks.append(.image(AnthropicImage(url: url.absoluteString)))
            case .data(let data, let mimeType):
                blocks.append(.image(AnthropicImage(base64Data: data.base64EncodedString(), mimeType: mimeType)))
            }
        }
    }
    return blocks
}

private struct AnthropicToolUse: Codable, Sendable {
    let type: String
    let id: String
    let name: String
    let input: [String: JSONValue]?

    init(id: String, name: String, input: [String: JSONValue]?) {
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
    }
}

private struct AnthropicToolResult: Codable, Sendable {
    let type: String
    let toolUseId: String
    let content: [AnthropicContent]

    enum CodingKeys: String, CodingKey {
        case type
        case toolUseId = "tool_use_id"
        case content
    }

    init(toolUseId: String, content: [AnthropicContent]) {
        self.type = "tool_result"
        self.toolUseId = toolUseId
        self.content = content
    }
}

private struct AnthropicMessageResponse: Codable, Sendable {
    let id: String
    let type: String
    let role: String
    let content: [AnthropicContent]
    let model: String
    let stopReason: StopReason?

    enum CodingKeys: String, CodingKey {
        case id, type, role, content, model
        case stopReason = "stop_reason"
    }

    enum StopReason: String, Codable {
        case endTurn = "end_turn"
        case maxTokens = "max_tokens"
        case stopSequence = "stop_sequence"
        case toolUse = "tool_use"
        case pauseTurn = "pause_turn"
        case refusal = "refusal"
        case modelContextWindowExceeded = "model_context_window_exceeded"
    }
}

private struct AnthropicErrorResponse: Codable { let error: AnthropicErrorDetail }
private struct AnthropicErrorDetail: Codable {
    let type: String
    let message: String
}

// MARK: - Streaming Event Types

private enum AnthropicStreamEvent: Codable, Sendable {
    case messageStart(MessageStartEvent)
    case contentBlockStart(ContentBlockStartEvent)
    case contentBlockDelta(ContentBlockDeltaEvent)
    case contentBlockStop(ContentBlockStopEvent)
    case messageDelta(MessageDeltaEvent)
    case messageStop
    case ping
    case ignored

    enum CodingKeys: String, CodingKey { case type }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "message_start":
            self = .messageStart(try MessageStartEvent(from: decoder))
        case "content_block_start":
            self = .contentBlockStart(try ContentBlockStartEvent(from: decoder))
        case "content_block_delta":
            self = .contentBlockDelta(try ContentBlockDeltaEvent(from: decoder))
        case "content_block_stop":
            self = .contentBlockStop(try ContentBlockStopEvent(from: decoder))
        case "message_delta":
            self = .messageDelta(try MessageDeltaEvent(from: decoder))
        case "message_stop":
            self = .messageStop
        case "ping":
            self = .ping
        default:
            self = .ignored
        }
    }

    func encode(to encoder: any Encoder) throws {
        switch self {
        case .messageStart(let event): try event.encode(to: encoder)
        case .contentBlockStart(let event): try event.encode(to: encoder)
        case .contentBlockDelta(let event): try event.encode(to: encoder)
        case .contentBlockStop(let event): try event.encode(to: encoder)
        case .messageDelta(let event): try event.encode(to: encoder)
        case .messageStop:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("message_stop", forKey: .type)
        case .ping:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("ping", forKey: .type)
        case .ignored:
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("ignored", forKey: .type)
        }
    }

    struct MessageStartEvent: Codable, Sendable {
        let type: String
        let message: AnthropicMessageResponse
    }

    struct ContentBlockStartEvent: Codable, Sendable {
        let type: String
        let index: Int
        let contentBlock: ContentBlock

        enum CodingKeys: String, CodingKey {
            case type, index
            case contentBlock = "content_block"
        }

        struct ContentBlock: Codable, Sendable {
            let type: String
            let text: String?
            
            // Used by tool use content blocks.
            let id: String?
            let name: String?
            let input: [String: JSONValue]?
        }
    }

    struct ContentBlockDeltaEvent: Codable, Sendable {
        let type: String
        let index: Int
        let delta: Delta

        enum Delta: Codable, Sendable {
            case textDelta(TextDelta)
            case inputJsonDelta(InputJsonDelta)
            case thinkingDelta(ThinkingDelta)
            case signatureDelta(SignatureDelta)
            case ignored

            enum CodingKeys: String, CodingKey { case type }

            init(from decoder: any Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                let type = try container.decode(String.self, forKey: .type)

                switch type {
                case "text_delta":
                    self = .textDelta(try TextDelta(from: decoder))
                case "input_json_delta":
                    self = .inputJsonDelta(try InputJsonDelta(from: decoder))
                case "thinking_delta":
                    self = .thinkingDelta(try ThinkingDelta(from: decoder))
                case "signature_delta":
                    self = .signatureDelta(try SignatureDelta(from: decoder))
                default:
                    self = .ignored
                }
            }

            func encode(to encoder: any Encoder) throws {
                switch self {
                case .textDelta(let delta): try delta.encode(to: encoder)
                case .inputJsonDelta(let delta): try delta.encode(to: encoder)
                case .thinkingDelta(let delta): try delta.encode(to: encoder)
                case .signatureDelta(let delta): try delta.encode(to: encoder)
                case .ignored:
                    var container = encoder.container(keyedBy: CodingKeys.self)
                    try container.encode("ignored", forKey: .type)
                }
            }

            struct TextDelta: Codable, Sendable {
                let type: String
                let text: String
            }

            struct InputJsonDelta: Codable, Sendable {
                let type: String
                let partialJson: String

                enum CodingKeys: String, CodingKey {
                    case type
                    case partialJson = "partial_json"
                }
            }
            
            struct ThinkingDelta: Codable, SendableMetatype {
                let type: String
                let thinking: String
            }
            
            /// Cryptographic signature for a completed thinking block.
            /// 
            /// Emitted at the end of a thinking block, even when ``CustomGenerationOptions/Thinking/display`` is set to `omitted`.
            /// The signature must be preserved verbatim for thought to be recovered in the transcript. Otherwise the Claude API will throw out any text provided in thinking blocks.
            struct SignatureDelta: Codable, Sendable {
                let type: String
                let signature: String
            }
        }
    }

    struct ContentBlockStopEvent: Codable, Sendable {
        let type: String
        let index: Int
    }

    struct MessageDeltaEvent: Codable, Sendable {
        let type: String
        let delta: Delta

        struct Delta: Codable, Sendable {
            let stopReason: String?
            let stopSequence: String?

            enum CodingKeys: String, CodingKey {
                case stopReason = "stop_reason"
                case stopSequence = "stop_sequence"
            }
        }
    }
}
