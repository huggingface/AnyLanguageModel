import Foundation
import JSONSchema

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A language model that connects to OpenAI-compatible APIs.
///
/// Use this model to generate text using OpenAI's Chat Completions or Responses APIs.
/// You can specify a custom base URL to work with OpenAI-compatible services.
///
/// ```swift
/// let model = OpenAILanguageModel(
///     apiKey: "your-api-key",
///     model: "gpt-4"
/// )
/// ```
public struct OpenAILanguageModel: LanguageModel {
    /// The reason the model is unavailable.
    /// This model is always available.
    public typealias UnavailableReason = Never

    /// The default base URL for OpenAI's API.
    public static let defaultBaseURL = URL(string: "https://api.openai.com/v1/")!

    /// The OpenAI API variant to use.
    public enum APIVariant: Sendable {
        /// When selected, use the Chat Completions API.
        /// https://platform.openai.com/docs/api-reference/chat/create
        case chatCompletions

        /// When selected, use the Responses API.
        /// https://platform.openai.com/docs/api-reference/responses
        case responses
    }

    /// Custom generation options specific to OpenAI-compatible APIs.
    ///
    /// Use this type to pass additional parameters that are not part of the
    /// standard ``GenerationOptions``, such as sampling parameters, penalties,
    /// and vendor-specific extensions.
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: OpenAILanguageModel.self] = .init(
    ///     topP: 0.9,
    ///     frequencyPenalty: 0.5,
    ///     presencePenalty: 0.5,
    ///     stopSequences: ["END"]
    /// )
    /// ```
    ///
    /// - Important: Custom sampling parameters in this type are sent directly to the API
    ///   and do not override equivalent settings in ``GenerationOptions``. Both values
    ///   will be included in the request if set; the API determines which takes precedence.
    public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions, Codable {
        // MARK: - Sampling Parameters

        /// An alternative to sampling with temperature, called nucleus sampling.
        ///
        /// The model considers the results of the tokens with `topP` probability mass.
        /// So `0.1` means only the tokens comprising the top 10% probability mass
        /// are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        ///
        /// Range: `0.0` to `1.0`. Defaults to `1.0`.
        public var topP: Double?

        /// Number between `-2.0` and `2.0`. Positive values penalize new tokens based
        /// on their existing frequency in the text so far, decreasing the model's
        /// likelihood to repeat the same line verbatim.
        public var frequencyPenalty: Double?

        /// Number between `-2.0` and `2.0`. Positive values penalize new tokens based
        /// on whether they appear in the text so far, increasing the model's likelihood
        /// to talk about new topics.
        public var presencePenalty: Double?

        /// Up to 4 sequences where the API will stop generating further tokens.
        /// The returned text will not contain the stop sequence.
        ///
        /// Not supported with latest reasoning models (o3 and o4-mini).
        public var stopSequences: [String]?

        /// Modify the likelihood of specified tokens appearing in the completion.
        ///
        /// Maps token IDs to an associated bias value from `-100` to `100`.
        /// Values between `-1` and `1` should decrease or increase likelihood of selection;
        /// values like `-100` or `100` should result in a ban or exclusive selection.
        public var logitBias: [Int: Int]?

        /// If specified, the system will make a best effort to sample deterministically,
        /// such that repeated requests with the same `seed` and parameters should return
        /// the same result.
        ///
        /// Determinism is not guaranteed.
        public var seed: Int?

        // MARK: - Output Configuration

        /// Whether to return log probabilities of the output tokens.
        ///
        /// If `true`, returns the log probabilities of each output token.
        public var logprobs: Bool?

        /// An integer between `0` and `20` specifying the number of most likely tokens
        /// to return at each token position, each with an associated log probability.
        ///
        /// `logprobs` must be set to `true` if this parameter is used.
        public var topLogprobs: Int?

        /// How many chat completion choices to generate for each input message.
        ///
        /// Note that you will be charged based on the number of generated tokens
        /// across all choices. Keep `n` as `1` to minimize costs.
        ///
        /// Only applicable to Chat Completions API.
        public var numberOfCompletions: Int?

        /// Constrains the verbosity of the model's response.
        ///
        /// Lower values will result in more concise responses, while higher values
        /// will result in more verbose responses.
        public var verbosity: Verbosity?

        // MARK: - Reasoning Configuration

        /// Constrains effort on reasoning for reasoning models.
        ///
        /// Reducing reasoning effort can result in faster responses and fewer tokens
        /// used on reasoning in a response.
        public var reasoningEffort: ReasoningEffort?

        /// Configuration options for reasoning models (Responses API).
        ///
        /// Use this for `gpt-5` and `o-series` models to configure reasoning behavior.
        public var reasoning: ReasoningConfiguration?

        // MARK: - Tool Configuration

        /// Whether to allow the model to run tool calls in parallel.
        ///
        /// Defaults to `true`.
        public var parallelToolCalls: Bool?

        /// The maximum number of total calls to built-in tools that can be processed
        /// in a response.
        ///
        /// This maximum number applies across all built-in tool calls, not per
        /// individual tool. Any further attempts to call a tool by the model will
        /// be ignored.
        ///
        /// Only applicable to Responses API.
        public var maxToolCalls: Int?

        // MARK: - Service Configuration

        /// Specifies the processing type used for serving the request.
        public var serviceTier: ServiceTier?

        /// Whether to store the generated model response for later retrieval via API.
        ///
        /// Chat Completions defaults to `false`. Responses API defaults to `true`.
        public var store: Bool?

        /// Set of up to 16 key-value pairs that can be attached to an object.
        ///
        /// This can be useful for storing additional information about the object
        /// in a structured format, and querying for objects via API or the dashboard.
        ///
        /// Keys have a maximum length of 64 characters.
        /// Values have a maximum length of 512 characters.
        public var metadata: [String: String]?

        /// A stable identifier used to help detect users of your application
        /// that may be violating OpenAI's usage policies.
        ///
        /// The IDs should be a string that uniquely identifies each user.
        /// We recommend hashing their username or email address to avoid
        /// sending any identifying information.
        public var safetyIdentifier: String?

        /// Used by OpenAI to cache responses for similar requests to optimize
        /// your cache hit rates.
        public var promptCacheKey: String?

        /// The retention policy for the prompt cache.
        ///
        /// Set to `"24h"` to enable extended prompt caching, which keeps cached
        /// prefixes active for longer, up to a maximum of 24 hours.
        public var promptCacheRetention: String?

        // MARK: - Truncation

        /// The truncation strategy to use for the model response.
        ///
        /// Only applicable to Responses API.
        public var truncation: Truncation?

        // MARK: - Extra Body

        /// Additional parameters to include in the request body.
        ///
        /// These parameters are merged into the top-level request JSON,
        /// allowing you to pass vendor-specific options like `reasoning`
        /// for Grok models via OpenRouter, or any parameters not explicitly
        /// modeled in this type.
        public var extraBody: [String: JSONValue]?

        // MARK: - Nested Types

        /// The verbosity level for model responses.
        public enum Verbosity: String, Hashable, Codable, Sendable {
            /// Produces more concise responses.
            case low
            /// The default verbosity level.
            case medium
            /// Produces more verbose responses.
            case high
        }

        /// The reasoning effort level for reasoning models.
        public enum ReasoningEffort: String, Hashable, Codable, Sendable {
            /// No reasoning (supported by gpt-5.1).
            case none
            /// Minimal reasoning effort.
            case minimal
            /// Low reasoning effort.
            case low
            /// Medium reasoning effort (default for most models).
            case medium
            /// High reasoning effort.
            case high
        }

        /// Configuration options for reasoning models (Responses API).
        public struct ReasoningConfiguration: Hashable, Codable, Sendable {
            /// The reasoning effort level.
            public var effort: ReasoningEffort?

            /// Optional summary mode for reasoning output.
            ///
            /// When set, provides a summary of the reasoning process.
            public var summary: String?

            enum CodingKeys: String, CodingKey {
                case effort
                case summary
            }

            /// Creates a reasoning configuration.
            ///
            /// - Parameters:
            ///   - effort: The reasoning effort level.
            ///   - summary: Optional summary mode.
            public init(effort: ReasoningEffort? = nil, summary: String? = nil) {
                self.effort = effort
                self.summary = summary
            }
        }

        /// The service tier for request processing.
        public enum ServiceTier: String, Hashable, Codable, Sendable {
            /// Uses the service tier configured in the Project settings.
            case auto
            /// Standard pricing and performance.
            case `default`
            /// Flex processing tier.
            case flex
            /// Priority processing tier.
            case priority
        }

        /// The truncation strategy for model responses.
        public enum Truncation: String, Hashable, Codable, Sendable {
            /// If the input exceeds the model's context window size, truncate
            /// the response by dropping items from the beginning.
            case auto
            /// If the input size exceeds the context window, fail with a 400 error.
            case disabled
        }

        enum CodingKeys: String, CodingKey {
            case topP = "top_p"
            case frequencyPenalty = "frequency_penalty"
            case presencePenalty = "presence_penalty"
            case stopSequences = "stop"
            case logitBias = "logit_bias"
            case seed
            case logprobs
            case topLogprobs = "top_logprobs"
            case numberOfCompletions = "n"
            case verbosity
            case reasoningEffort = "reasoning_effort"
            case reasoning
            case parallelToolCalls = "parallel_tool_calls"
            case maxToolCalls = "max_tool_calls"
            case serviceTier = "service_tier"
            case store
            case metadata
            case safetyIdentifier = "safety_identifier"
            case promptCacheKey = "prompt_cache_key"
            case promptCacheRetention = "prompt_cache_retention"
            case truncation
            case extraBody = "extra_body"
        }

        /// Creates custom generation options for OpenAI-compatible APIs.
        ///
        /// - Parameters:
        ///   - topP: Nucleus sampling probability threshold.
        ///   - frequencyPenalty: Penalty for token frequency (-2.0 to 2.0).
        ///   - presencePenalty: Penalty for token presence (-2.0 to 2.0).
        ///   - stopSequences: Up to 4 sequences that stop generation.
        ///   - logitBias: Token ID to bias value mapping.
        ///   - seed: Seed for deterministic sampling.
        ///   - logprobs: Whether to return log probabilities.
        ///   - topLogprobs: Number of most likely tokens to return (0-20).
        ///   - numberOfCompletions: Number of completions to generate.
        ///   - verbosity: Response verbosity level.
        ///   - reasoningEffort: Reasoning effort for reasoning models.
        ///   - reasoning: Reasoning configuration (Responses API).
        ///   - parallelToolCalls: Whether to allow parallel tool calls.
        ///   - maxToolCalls: Maximum number of tool calls (Responses API).
        ///   - serviceTier: Service tier for request processing.
        ///   - store: Whether to store the response.
        ///   - metadata: Key-value pairs for additional information.
        ///   - safetyIdentifier: User identifier for safety detection.
        ///   - promptCacheKey: Key for response caching.
        ///   - promptCacheRetention: Cache retention policy.
        ///   - truncation: Truncation strategy (Responses API).
        ///   - extraBody: Additional parameters for the request body.
        public init(
            topP: Double? = nil,
            frequencyPenalty: Double? = nil,
            presencePenalty: Double? = nil,
            stopSequences: [String]? = nil,
            logitBias: [Int: Int]? = nil,
            seed: Int? = nil,
            logprobs: Bool? = nil,
            topLogprobs: Int? = nil,
            numberOfCompletions: Int? = nil,
            verbosity: Verbosity? = nil,
            reasoningEffort: ReasoningEffort? = nil,
            reasoning: ReasoningConfiguration? = nil,
            parallelToolCalls: Bool? = nil,
            maxToolCalls: Int? = nil,
            serviceTier: ServiceTier? = nil,
            store: Bool? = nil,
            metadata: [String: String]? = nil,
            safetyIdentifier: String? = nil,
            promptCacheKey: String? = nil,
            promptCacheRetention: String? = nil,
            truncation: Truncation? = nil,
            extraBody: [String: JSONValue]? = nil
        ) {
            self.topP = topP
            self.frequencyPenalty = frequencyPenalty
            self.presencePenalty = presencePenalty
            self.stopSequences = stopSequences
            self.logitBias = logitBias
            self.seed = seed
            self.logprobs = logprobs
            self.topLogprobs = topLogprobs
            self.numberOfCompletions = numberOfCompletions
            self.verbosity = verbosity
            self.reasoningEffort = reasoningEffort
            self.reasoning = reasoning
            self.parallelToolCalls = parallelToolCalls
            self.maxToolCalls = maxToolCalls
            self.serviceTier = serviceTier
            self.store = store
            self.metadata = metadata
            self.safetyIdentifier = safetyIdentifier
            self.promptCacheKey = promptCacheKey
            self.promptCacheRetention = promptCacheRetention
            self.truncation = truncation
            self.extraBody = extraBody
        }
    }

    /// The base URL for the API endpoint.
    public let baseURL: URL

    /// The closure providing the API key for authentication.
    private let tokenProvider: @Sendable () -> String

    /// The model identifier to use for generation.
    public let model: String

    /// The API variant to use.
    public let apiVariant: APIVariant

    private let httpSession: HTTPSession

    /// Creates an OpenAI language model.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the API endpoint. Defaults to OpenAI's official API.
    ///   - apiKey: Your OpenAI API key or a closure that returns it.
    ///   - model: The model identifier (for example, "gpt-4" or "gpt-3.5-turbo").
    ///   - apiVariant: The API variant to use. Defaults to `.chatCompletions`.
    ///   - session: The HTTP session or client used for network requests.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        model: String,
        apiVariant: APIVariant = .chatCompletions,
        session: HTTPSession = makeDefaultSession(),
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.model = model
        self.apiVariant = apiVariant
        self.httpSession = session
    }

    public func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        // Convert tools if any are available in the session
        let openAITools: [OpenAITool]? = {
            guard !session.tools.isEmpty else { return nil }
            var converted: [OpenAITool] = []
            converted.reserveCapacity(session.tools.count)
            for tool in session.tools {
                converted.append(convertToolToOpenAIFormat(tool))
            }
            return converted
        }()

        switch apiVariant {
        case .chatCompletions:
            return try await respondWithChatCompletions(
                messages: session.transcript.toOpenAIMessages(),
                tools: openAITools,
                generating: type,
                options: options,
                session: session
            )
        case .responses:
            return try await respondWithResponses(
                messages: session.transcript.toOpenAIMessages(),
                tools: openAITools,
                generating: type,
                options: options,
                session: session
            )
        }
    }

    private func respondWithChatCompletions<Content>(
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        generating type: Content.Type,
        options: GenerationOptions,
        session: LanguageModelSession
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {

        var entries: [Transcript.Entry] = []
        var text = ""
        var messages = messages

        // Loop until no more tool calls
        while true {
            let params = try ChatCompletions.createRequestBody(
                model: model,
                messages: messages,
                tools: tools,
                generating: type,
                options: options,
                stream: false
            )

            let url = baseURL.appendingPathComponent("chat/completions")
            let body = try JSONEncoder().encode(params)
            let resp: ChatCompletions.Response = try await httpSession.fetch(
                .post,
                url: url,
                headers: [
                    "Authorization": "Bearer \(tokenProvider())"
                ],
                body: body
            )

            guard let choice = resp.choices.first else {
                throw OpenAILanguageModelError.noResponseGenerated
            }

            if let refusalMessage = choice.message.refusal {
                let refusalEntry = Transcript.Entry.response(
                    Transcript.Response(assetIDs: [], segments: [.text(.init(content: refusalMessage))])
                )
                throw LanguageModelSession.GenerationError.refusal(
                    LanguageModelSession.GenerationError.Refusal(transcriptEntries: [refusalEntry]),
                    LanguageModelSession.GenerationError.Context(
                        debugDescription: "OpenAI model refused to generate response: \(refusalMessage)"
                    )
                )
            }

            let toolCallMessage = choice.message
            if let toolCalls = toolCallMessage.toolCalls, !toolCalls.isEmpty {
                if let value = try? JSONValue(toolCallMessage) {
                    messages.append(OpenAIMessage(role: .raw(rawContent: value), content: .text("")))
                }
                let resolution = try await resolveToolCalls(toolCalls, session: session)
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
                        entries.append(.toolCalls(Transcript.ToolCalls(invocations.map { $0.call })))
                        for invocation in invocations {
                            let output = invocation.output
                            entries.append(.toolOutput(output))
                            messages.append(
                                OpenAIMessage(
                                    role: .tool(id: invocation.call.id),
                                    content: .text(convertSegmentsToToolContentString(output.segments))
                                )
                            )
                        }
                        continue
                    }
                }
            }

            text = choice.message.content ?? ""
            break
        }

        if type == String.self {
            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(entries)
            )
        }

        let generatedContent = try GeneratedContent(json: text)
        let content = try type.init(generatedContent)
        return LanguageModelSession.Response(
            content: content,
            rawContent: generatedContent,
            transcriptEntries: ArraySlice(entries)
        )
    }

    private func respondWithResponses<Content>(
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        generating type: Content.Type,
        options: GenerationOptions,
        session: LanguageModelSession
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        var entries: [Transcript.Entry] = []
        var text = ""
        var lastOutput: [JSONValue]?
        var messages = messages

        let url = baseURL.appendingPathComponent("responses")

        // Loop until no more tool calls
        while true {
            let params = try Responses.createRequestBody(
                model: model,
                messages: messages,
                tools: tools,
                generating: type,
                options: options,
                stream: false
            )

            let encoder = JSONEncoder()
            let body = try encoder.encode(params)
            let resp: Responses.Response = try await httpSession.fetch(
                .post,
                url: url,
                headers: [
                    "Authorization": "Bearer \(tokenProvider())"
                ],
                body: body
            )

            let toolCalls = extractToolCallsFromOutput(resp.output)
            lastOutput = resp.output
            if !toolCalls.isEmpty {
                if let output = resp.output {
                    for msg in output {
                        messages.append(OpenAIMessage(role: .raw(rawContent: msg), content: .text("")))
                    }
                }
                let resolution = try await resolveToolCalls(toolCalls, session: session)
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
                        entries.append(.toolCalls(Transcript.ToolCalls(invocations.map { $0.call })))

                        for invocation in invocations {
                            let output = invocation.output
                            entries.append(.toolOutput(output))
                            messages.append(
                                OpenAIMessage(
                                    role: .tool(id: invocation.call.id),
                                    content: .text(convertSegmentsToToolContentString(output.segments))
                                )
                            )
                        }
                        continue
                    }
                }
            }

            text = resp.outputText ?? extractTextFromOutput(resp.output) ?? ""

            break
        }

        if type == String.self {
            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(entries)
            )
        }

        if let jsonString = extractJSONFromOutput(lastOutput) {
            let generatedContent = try GeneratedContent(json: jsonString)
            let content = try type.init(generatedContent)
            return LanguageModelSession.Response(
                content: content,
                rawContent: generatedContent,
                transcriptEntries: ArraySlice(entries)
            )
        }
        throw OpenAILanguageModelError.noResponseGenerated
    }

    public func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        // Convert tools if any are available in the session
        let openAITools: [OpenAITool]? = {
            guard !session.tools.isEmpty else { return nil }
            var converted: [OpenAITool] = []
            converted.reserveCapacity(session.tools.count)
            for tool in session.tools {
                converted.append(convertToolToOpenAIFormat(tool))
            }
            return converted
        }()

        switch apiVariant {
        case .responses:
            let url = baseURL.appendingPathComponent("responses")

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                do {
                    let params = try Responses.createRequestBody(
                        model: model,
                        messages: session.transcript.toOpenAIMessages(),
                        tools: openAITools,
                        generating: type,
                        options: options,
                        stream: true
                    )
                    let task = Task { @Sendable in
                        do {
                            let body = try JSONEncoder().encode(params)

                            let events: AsyncThrowingStream<OpenAIResponsesServerEvent, any Error> =
                                httpSession.fetchEventStream(
                                    .post,
                                    url: url,
                                    headers: [
                                        "Authorization": "Bearer \(tokenProvider())"
                                    ],
                                    body: body
                                )

                            var accumulatedText = ""

                            for try await event in events {
                                switch event {
                                case .outputTextDelta(let delta):
                                    accumulatedText += delta

                                    var raw: GeneratedContent
                                    let content: Content.PartiallyGenerated?

                                    if type == String.self {
                                        raw = GeneratedContent(accumulatedText)
                                        content = (accumulatedText as! Content).asPartiallyGenerated()
                                    } else {
                                        raw =
                                            (try? GeneratedContent(json: accumulatedText))
                                            ?? GeneratedContent(accumulatedText)
                                        if let parsed = try? type.init(raw) {
                                            content = parsed.asPartiallyGenerated()
                                        } else {
                                            // Skip snapshots until the accumulated JSON parses.
                                            content = nil
                                        }
                                    }

                                    if let content {
                                        continuation.yield(.init(content: content, rawContent: raw))
                                    }

                                case .toolCallCreated(_):
                                    // Minimal streaming implementation ignores tool call events
                                    break
                                case .toolCallDelta(_):
                                    // Minimal streaming implementation ignores tool call deltas
                                    break
                                case .completed(_):
                                    continuation.finish()
                                case .ignored:
                                    break
                                }
                            }

                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    continuation.onTermination = { _ in task.cancel() }
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            return LanguageModelSession.ResponseStream(stream: stream)

        case .chatCompletions:
            let url = baseURL.appendingPathComponent("chat/completions")

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                do {
                    let params = try ChatCompletions.createRequestBody(
                        model: model,
                        messages: session.transcript.toOpenAIMessages(),
                        tools: openAITools,
                        generating: type,
                        options: options,
                        stream: true
                    )

                    let task = Task { @Sendable in
                        do {
                            let body = try JSONEncoder().encode(params)

                            let events: AsyncThrowingStream<OpenAIChatCompletionsChunk, any Error> =
                                httpSession.fetchEventStream(
                                    .post,
                                    url: url,
                                    headers: [
                                        "Authorization": "Bearer \(tokenProvider())"
                                    ],
                                    body: body
                                )

                            var accumulatedText = ""

                            for try await chunk in events {
                                if let choice = chunk.choices.first {
                                    if let piece = choice.delta.content, !piece.isEmpty {
                                        accumulatedText += piece

                                        var raw: GeneratedContent
                                        let content: Content.PartiallyGenerated?

                                        if type == String.self {
                                            raw = GeneratedContent(accumulatedText)
                                            content = (accumulatedText as! Content).asPartiallyGenerated()
                                        } else {
                                            raw =
                                                (try? GeneratedContent(json: accumulatedText))
                                                ?? GeneratedContent(accumulatedText)
                                            if let parsed = try? type.init(raw) {
                                                content = parsed.asPartiallyGenerated()
                                            } else {
                                                // Skip snapshots until the accumulated JSON parses.
                                                content = nil
                                            }
                                        }

                                        if let content {
                                            continuation.yield(.init(content: content, rawContent: raw))
                                        }
                                    }

                                    if choice.finishReason != nil {
                                        continuation.finish()
                                    }
                                }
                            }

                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    continuation.onTermination = { _ in task.cancel() }
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            return LanguageModelSession.ResponseStream(stream: stream)
        }
    }
}

// MARK: - API Variants

private enum ChatCompletions {
    static func createRequestBody<Content: Generable>(
        model: String,
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        generating type: Content.Type,
        options: GenerationOptions,
        stream: Bool
    ) throws -> JSONValue {
        var body: [String: JSONValue] = [
            "model": .string(model),
            "messages": .array(messages.map { $0.jsonValue(for: .chatCompletions) }),
            "stream": .bool(stream),
        ]

        if let tools {
            body["tools"] = .array(tools.map { $0.jsonValue(for: .chatCompletions) })
        }

        if type != String.self {
            let jsonSchemaValue = try type.generationSchema.toJSONValueForOpenAIStrictMode()
            body["response_format"] = .object([
                "type": .string("json_schema"),
                "json_schema": .object([
                    "name": .string("response_schema"),
                    "strict": .bool(true),
                    "schema": jsonSchemaValue,
                ]),
            ])
        }

        if let temperature = options.temperature {
            body["temperature"] = .double(temperature)
        }
        if let maxTokens = options.maximumResponseTokens {
            body["max_completion_tokens"] = .int(maxTokens)
        }

        // Apply custom options
        if let customOptions = options[custom: OpenAILanguageModel.self] {
            // Sampling parameters
            if let topP = customOptions.topP {
                body["top_p"] = .double(topP)
            }
            if let frequencyPenalty = customOptions.frequencyPenalty {
                body["frequency_penalty"] = .double(frequencyPenalty)
            }
            if let presencePenalty = customOptions.presencePenalty {
                body["presence_penalty"] = .double(presencePenalty)
            }
            if let stopSequences = customOptions.stopSequences, !stopSequences.isEmpty {
                body["stop"] = .array(stopSequences.map { .string($0) })
            }
            if let logitBias = customOptions.logitBias, !logitBias.isEmpty {
                body["logit_bias"] = .object(
                    Dictionary(uniqueKeysWithValues: logitBias.map { (String($0.key), JSONValue.int($0.value)) })
                )
            }
            if let seed = customOptions.seed {
                body["seed"] = .int(seed)
            }

            // Output configuration
            if let logprobs = customOptions.logprobs {
                body["logprobs"] = .bool(logprobs)
            }
            if let topLogprobs = customOptions.topLogprobs {
                body["top_logprobs"] = .int(topLogprobs)
            }
            if let n = customOptions.numberOfCompletions {
                body["n"] = .int(n)
            }
            if let verbosity = customOptions.verbosity {
                body["verbosity"] = .string(verbosity.rawValue)
            }

            // Reasoning configuration
            if let reasoningEffort = customOptions.reasoningEffort {
                body["reasoning_effort"] = .string(reasoningEffort.rawValue)
            }

            // Tool configuration
            if let parallelToolCalls = customOptions.parallelToolCalls {
                body["parallel_tool_calls"] = .bool(parallelToolCalls)
            }

            // Service configuration
            if let serviceTier = customOptions.serviceTier {
                body["service_tier"] = .string(serviceTier.rawValue)
            }
            if let store = customOptions.store {
                body["store"] = .bool(store)
            }
            if let metadata = customOptions.metadata, !metadata.isEmpty {
                body["metadata"] = .object(
                    Dictionary(uniqueKeysWithValues: metadata.map { ($0.key, JSONValue.string($0.value)) })
                )
            }
            if let safetyIdentifier = customOptions.safetyIdentifier {
                body["safety_identifier"] = .string(safetyIdentifier)
            }
            if let promptCacheKey = customOptions.promptCacheKey {
                body["prompt_cache_key"] = .string(promptCacheKey)
            }
            if let promptCacheRetention = customOptions.promptCacheRetention {
                body["prompt_cache_retention"] = .string(promptCacheRetention)
            }

            // Merge extraBody last to allow overrides
            if let extraBody = customOptions.extraBody {
                for (key, value) in extraBody {
                    body[key] = value
                }
            }
        }

        return .object(body)
    }

    struct Response: Decodable, Sendable {
        let id: String
        let choices: [Choice]

        struct Choice: Codable, Sendable {
            let message: Message
            let finishReason: String?

            private enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }

        struct Message: Codable, Sendable {
            let role: String
            let content: String?
            let refusal: String?
            let toolCalls: [OpenAIToolCall]?

            private enum CodingKeys: String, CodingKey {
                case role
                case content
                case refusal
                case toolCalls = "tool_calls"
            }
        }
    }
}

private enum Responses {
    static func createRequestBody<Content: Generable>(
        model: String,
        messages: [OpenAIMessage],
        tools: [OpenAITool]?,
        generating type: Content.Type,
        options: GenerationOptions,
        stream: Bool
    ) throws -> JSONValue {
        // Build input blocks from the user message content

        var body: [String: JSONValue] = [
            "model": .string(model),
            "stream": .bool(stream),
        ]

        var outputs: [JSONValue] = []
        for msg in messages {
            switch msg.role {
            case .user:

                let userMessage = msg
                // Wrap user content into a single top-level message as required by Responses API
                let contentBlocks: [JSONValue]
                switch userMessage.content {
                case .text(let text):
                    contentBlocks = [
                        .object(["type": .string("input_text"), "text": .string(text)])
                    ]
                case .blocks(let blocks):
                    contentBlocks = blocks.map { block in
                        switch block {
                        case .text(let text):
                            return .object(["type": .string("input_text"), "text": .string(text)])
                        case .imageURL(let url):
                            return .object([
                                "type": .string("input_image"),
                                "image_url": .string(url),
                            ])
                        }
                    }
                }
                let object = JSONValue.object([
                    "type": .string("message"),
                    "role": .string("user"),
                    "content": .array(contentBlocks),
                ])
                outputs.append(object)

            case .tool(let id):
                let outputValue: JSONValue
                switch msg.content {
                case .text(let text):
                    outputValue = .string(text)
                case .blocks(let blocks):
                    outputValue = .array(
                        blocks.map { block in
                            switch block {
                            case .text(let text):
                                return .object(["type": .string("input_text"), "text": .string(text)])
                            case .imageURL(let url):
                                return .object([
                                    "type": .string("input_image"),
                                    "image_url": .string(url),
                                ])
                            }
                        }
                    )
                }
                outputs.append(
                    .object([
                        "type": .string("function_call_output"),
                        "call_id": .string(id),
                        "output": outputValue,
                    ])
                )

            case .raw(rawContent: let rawContent):
                // Convert Chat Completions assistant+tool_calls to Responses API function_call items
                if case .object(let assistantMessageObject) = rawContent,
                    case .string(let messageRole) = assistantMessageObject["role"],
                    messageRole == "assistant",
                    case .array(let assistantToolCalls) = assistantMessageObject["tool_calls"]
                {
                    for assistantToolCall in assistantToolCalls {
                        if case .object(let toolCallObject) = assistantToolCall,
                            case .string(let toolCallID) = toolCallObject["id"],
                            case .object(let functionCallObject) = toolCallObject["function"],
                            case .string(let functionName) = functionCallObject["name"],
                            case .string(let functionArguments) = functionCallObject["arguments"]
                        {
                            outputs.append(
                                .object([
                                    "type": .string("function_call"),
                                    "call_id": .string(toolCallID),
                                    "name": .string(functionName),
                                    "arguments": .string(functionArguments),
                                ])
                            )
                        }
                    }
                } else {
                    outputs.append(rawContent)
                }

            case .system:
                let systemMessage = msg
                switch systemMessage.content {
                case .text(let text):
                    body["instructions"] = .string(text)
                case .blocks(let blocks):
                    // Concatenate text blocks for instructions; ignore images
                    let text = blocks.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined(
                        separator: "\n"
                    )
                    if !text.isEmpty { body["instructions"] = .string(text) }
                }

            case .assistant:
                break
            }
        }
        body["input"] = .array(outputs)

        if let tools {
            body["tools"] = .array(tools.map { $0.jsonValue(for: .responses) })
        }

        if type != String.self {
            let jsonSchemaValue = try type.generationSchema.toJSONValueForOpenAIStrictMode()
            body["text"] = .object([
                "format": .object([
                    "type": .string("json_schema"),
                    "name": .string("response_schema"),
                    "strict": .bool(true),
                    "schema": jsonSchemaValue,
                ])
            ])
        }

        if let temperature = options.temperature {
            body["temperature"] = .double(temperature)
        }
        if let maxTokens = options.maximumResponseTokens {
            body["max_output_tokens"] = .int(maxTokens)
        }

        // Apply custom options
        if let customOptions = options[custom: OpenAILanguageModel.self] {
            // Sampling parameters
            if let topP = customOptions.topP {
                body["top_p"] = .double(topP)
            }

            // Output configuration
            if let topLogprobs = customOptions.topLogprobs {
                body["top_logprobs"] = .int(topLogprobs)
            }

            // Reasoning configuration
            if let reasoning = customOptions.reasoning {
                var reasoningObj: [String: JSONValue] = [:]
                if let effort = reasoning.effort {
                    reasoningObj["effort"] = .string(effort.rawValue)
                }
                if let summary = reasoning.summary {
                    reasoningObj["summary"] = .string(summary)
                }
                if !reasoningObj.isEmpty {
                    body["reasoning"] = .object(reasoningObj)
                }
            }

            // Tool configuration
            if let parallelToolCalls = customOptions.parallelToolCalls {
                body["parallel_tool_calls"] = .bool(parallelToolCalls)
            }
            if let maxToolCalls = customOptions.maxToolCalls {
                body["max_tool_calls"] = .int(maxToolCalls)
            }

            // Service configuration
            if let serviceTier = customOptions.serviceTier {
                body["service_tier"] = .string(serviceTier.rawValue)
            }
            if let store = customOptions.store {
                body["store"] = .bool(store)
            }
            if let metadata = customOptions.metadata, !metadata.isEmpty {
                body["metadata"] = .object(
                    Dictionary(uniqueKeysWithValues: metadata.map { ($0.key, JSONValue.string($0.value)) })
                )
            }
            if let safetyIdentifier = customOptions.safetyIdentifier {
                body["safety_identifier"] = .string(safetyIdentifier)
            }
            if let promptCacheKey = customOptions.promptCacheKey {
                body["prompt_cache_key"] = .string(promptCacheKey)
            }
            if let promptCacheRetention = customOptions.promptCacheRetention {
                body["prompt_cache_retention"] = .string(promptCacheRetention)
            }

            // Truncation
            if let truncation = customOptions.truncation {
                body["truncation"] = .string(truncation.rawValue)
            }

            // Merge extraBody last to allow overrides
            if let extraBody = customOptions.extraBody {
                for (key, value) in extraBody {
                    body[key] = value
                }
            }
        }

        return .object(body)
    }

    struct Response: Decodable, Sendable {
        let id: String
        let output: [JSONValue]?
        let error: [JSONValue]?
        let outputText: String?
        let finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case id
            case output
            case outputText = "output_text"
            case finishReason = "finish_reason"
            case error = "error"
        }
    }
}

// MARK: - Supporting Types

extension Transcript {
    fileprivate func toOpenAIMessages() -> [OpenAIMessage] {
        var messages = [OpenAIMessage]()
        for item in self {
            switch item {
            case .instructions(let instructions):
                messages.append(
                    .init(
                        role: .system,
                        content: .blocks(convertSegmentsToOpenAIBlocks(instructions.segments))
                    )
                )
            case .prompt(let prompt):
                messages.append(
                    .init(
                        role: .user,
                        content: .blocks(convertSegmentsToOpenAIBlocks(prompt.segments))
                    )
                )
            case .response(let response):
                messages.append(
                    .init(
                        role: .assistant,
                        content: .blocks(convertSegmentsToOpenAIBlocks(response.segments))
                    )
                )
            case .toolCalls(let toolCalls):
                // Add assistant message with tool calls
                let openAIToolCalls: [JSONValue] = toolCalls.map { call in
                    let argumentsJSON: String
                    if let data = try? JSONEncoder().encode(call.arguments),
                        let jsonString = String(data: data, encoding: .utf8)
                    {
                        argumentsJSON = jsonString
                    } else {
                        argumentsJSON = "{}"
                    }

                    return .object([
                        "id": .string(call.id),
                        "type": .string("function"),
                        "function": .object([
                            "name": .string(call.toolName),
                            "arguments": .string(argumentsJSON),
                        ]),
                    ])
                }

                let rawMessage: JSONValue = .object([
                    "role": .string("assistant"),
                    "content": .null,
                    "tool_calls": .array(openAIToolCalls),
                ])

                messages.append(
                    .init(
                        role: .raw(rawContent: rawMessage),
                        content: .text("")
                    )
                )
            case .toolOutput(let toolOutput):
                messages.append(
                    .init(
                        role: .tool(id: toolOutput.id),
                        content: .text(convertSegmentsToToolContentString(toolOutput.segments))
                    )
                )
            }
        }
        return messages
    }
}

private struct OpenAIMessage: Hashable, Codable, Sendable {
    enum Role: Hashable, Codable, Sendable {
        case system, user, assistant, raw(rawContent: JSONValue), tool(id: String)

        var description: String {
            switch self {
            case .system: return "system"
            case .user: return "user"
            case .assistant: return "assistant"
            case .tool(id: _): return "tool"
            case .raw(rawContent: _): return "raw"
            }
        }
    }

    enum Content: Hashable, Codable, Sendable {
        case text(String)
        case blocks([Block])
    }

    let role: Role
    let content: Content

    func contentAsJsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {
        switch content {
        case .text(let text):
            switch apiVariant {
            case .chatCompletions:
                return .string(text)
            case .responses:
                return .array([.object(["type": .string("text"), "text": .string(text)])])
            }
        case .blocks(let blocks):
            switch apiVariant {
            case .chatCompletions:
                return .array(blocks.map { $0.jsonValueForChatCompletions })
            case .responses:
                return .array(blocks.map { $0.jsonValueForResponses })
            }
        }
    }

    func jsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {
        switch role {
        case .raw(rawContent: let rawContent):
            return rawContent

        case .tool(id: let id):
            switch apiVariant {
            case .chatCompletions:
                return .object([
                    "role": .string(role.description),
                    "tool_call_id": .string(id),
                    "content": contentAsJsonValue(for: apiVariant),
                ])
            case .responses:
                return .object([
                    "type": .string("function_call_output"),
                    "call_id": .string(id),
                    "content": contentAsJsonValue(for: apiVariant),
                ])
            }

        case .system, .user, .assistant:
            return .object([
                "role": .string(role.description),
                "content": contentAsJsonValue(for: apiVariant),
            ])
        }
    }

}

private enum Block: Hashable, Codable, Sendable {
    case text(String)
    case imageURL(String)

    var jsonValueForChatCompletions: JSONValue {
        switch self {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])
        case .imageURL(let url):
            return .object([
                "type": .string("image_url"),
                "image_url": .object(["url": .string(url)]),
            ])
        }
    }

    var jsonValueForResponses: JSONValue {
        switch self {
        case .text(let text):
            return .object(["type": .string("text"), "text": .string(text)])
        case .imageURL(let url):
            // Responses API uses input_image at top-level input, but inside messages we mirror block
            return .object([
                "type": .string("input_image"),
                "image_url": .object(["url": .string(url)]),
            ])
        }
    }
}

private func convertSegmentsToOpenAIBlocks(_ segments: [Transcript.Segment]) -> [Block] {
    var blocks: [Block] = []
    blocks.reserveCapacity(segments.count)
    for segment in segments {
        switch segment {
        case .text(let text):
            blocks.append(.text(text.content))
        case .structure(let structured):
            switch structured.content.kind {
            case .string(let text):
                blocks.append(.text(text))
            default:
                blocks.append(.text(structured.content.jsonString))
            }
        case .image(let image):
            switch image.source {
            case .url(let url):
                blocks.append(.imageURL(url.absoluteString))
            case .data(let data, let mimeType):
                let b64 = data.base64EncodedString()
                let dataURL = "data:\(mimeType);base64,\(b64)"
                blocks.append(.imageURL(dataURL))
            }
        }
    }
    return blocks
}

/// Converts a list of transcript segments to a string representation suitable for tool message content.
private func convertSegmentsToToolContentString(_ segments: [Transcript.Segment]) -> String {
    // Tool message content must be string (or text-only parts) per API spec.
    // String is used for broad provider compatibility.
    // Image segments are intentionally omitted because tool outputs only support text.
    segments.compactMap { segment in
        switch segment {
        case .text(let textSegment):
            return textSegment.content
        case .structure(let structuredSegment):
            switch structuredSegment.content.kind {
            case .string(let text): return text
            default: return structuredSegment.content.jsonString
            }
        case .image:
            return nil
        }
    }.joined(separator: "\n")
}

private struct OpenAITool: Hashable, Codable, Sendable {
    let type: String
    let function: OpenAIFunction

    func jsonValue(for apiVariant: OpenAILanguageModel.APIVariant) -> JSONValue {
        switch apiVariant {
        case .chatCompletions:
            return .object([
                "type": .string(type),
                "function": function.jsonValue,
            ])
        case .responses:
            // Responses API expects name, description, and parameters at the top level
            var obj: [String: JSONValue] = [
                "type": .string(type),
                "name": .string(function.name),
                "description": .string(function.description),
            ]
            if let rawParameters = function.rawParameters {
                obj["parameters"] = rawParameters
            } else if let parameters = function.parameters {
                obj["parameters"] = parameters.jsonValue
            }
            return .object(obj)
        }
    }
}

private struct OpenAIFunction: Hashable, Codable, Sendable {
    let name: String
    let description: String
    let parameters: OpenAIParameters?
    // When available, prefer passing raw JSON Schema converted from GenerationSchema
    // to preserve nested object structures.
    let rawParameters: JSONValue?

    var jsonValue: JSONValue {
        var obj: [String: JSONValue] = [
            "name": .string(name),
            "description": .string(description),
        ]
        if let rawParameters {
            obj["parameters"] = rawParameters
        } else if let parameters {
            obj["parameters"] = parameters.jsonValue
        }
        return .object(obj)
    }
}

private struct OpenAIParameters: Hashable, Codable, Sendable {
    let type: String
    let properties: [String: OpenAISchema]
    let required: [String]

    var jsonValue: JSONValue {
        return .object([
            "type": .string(type),
            "properties": .object(properties.mapValues { $0.jsonValue }),
            "required": .array(required.map { .string($0) }),
        ])
    }
}

private struct OpenAISchema: Hashable, Codable, Sendable {
    let type: String
    let description: String?
    let enumValues: [String]?

    var jsonValue: JSONValue {
        var obj: [String: JSONValue] = ["type": .string(type)]
        if let description { obj["description"] = .string(description) }
        if let enumValues { obj["enum"] = .array(enumValues.map { .string($0) }) }
        return .object(obj)
    }
}

private struct OpenAIToolCall: Codable, Sendable {
    let id: String?
    let type: String?
    let function: OpenAIToolFunction?
}

private struct OpenAIToolFunction: Codable, Sendable {
    let name: String
    let arguments: String?
}

private enum OpenAIResponsesServerEvent: Decodable, Sendable {
    case outputTextDelta(String)
    case toolCallCreated(OpenAIToolCall)
    case toolCallDelta(OpenAIToolCall)
    case completed(String)
    case ignored

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decodeIfPresent(String.self, forKey: .type)
        switch type {
        case "response.output_text.delta":
            self = .outputTextDelta(try container.decode(String.self, forKey: .delta))
        case "response.tool_call.created":
            self = .toolCallCreated(try container.decode(OpenAIToolCall.self, forKey: .toolCall))
        case "response.tool_call.delta":
            self = .toolCallDelta(try container.decode(OpenAIToolCall.self, forKey: .toolCall))
        case "response.completed":
            self = .completed((try? container.decode(String.self, forKey: .finishReason)) ?? "stop")
        default:
            self = .ignored
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case delta
        case toolCall = "tool_call"
        case finishReason = "finish_reason"
    }
}

private struct OpenAIChatCompletionsChunk: Decodable, Sendable {
    struct Choice: Decodable, Sendable {
        struct Delta: Decodable, Sendable {
            let role: String?
            let content: String?
        }
        let delta: Delta
        let finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let choices: [Choice]
}

private struct OpenAIToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private enum OpenAIToolResolutionOutcome {
    case stop(calls: [Transcript.ToolCall])
    case invocations([OpenAIToolInvocationResult])
}

private func resolveToolCalls(
    _ toolCalls: [OpenAIToolCall],
    session: LanguageModelSession
) async throws -> OpenAIToolResolutionOutcome {
    if toolCalls.isEmpty { return .invocations([]) }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var transcriptCalls: [Transcript.ToolCall] = []
    transcriptCalls.reserveCapacity(toolCalls.count)
    for call in toolCalls {
        guard let function = call.function else { continue }
        let args = try toGeneratedContent(function.arguments)
        let callID = call.id ?? UUID().uuidString
        transcriptCalls.append(
            Transcript.ToolCall(
                id: callID,
                toolName: function.name,
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

    var results: [OpenAIToolInvocationResult] = []
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
            results.append(OpenAIToolInvocationResult(call: call, output: output))
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
                results.append(OpenAIToolInvocationResult(call: call, output: output))
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
                results.append(OpenAIToolInvocationResult(call: call, output: output))
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

// MARK: - Converters

private func convertToolToOpenAIFormat(_ tool: any Tool) -> OpenAITool {
    // Prefer passing through a JSONSchema value built from GenerationSchema
    // where possible; fallback to minimal type/required map.
    let rawParameters: JSONValue?

    // Handle the case where the schema has a root reference
    if let resolvedSchema = tool.parameters.withResolvedRoot() {
        rawParameters = try? JSONValue(resolvedSchema)
    } else {
        rawParameters = try? JSONValue(tool.parameters)
    }

    let fn = OpenAIFunction(
        name: tool.name,
        description: tool.description,
        parameters: nil,
        rawParameters: rawParameters
    )
    return OpenAITool(type: "function", function: fn)
}

private func emptyResponseContent<Content: Generable>(
    for type: Content.Type
) throws -> (content: Content, rawContent: GeneratedContent) {
    if type == String.self {
        let raw = GeneratedContent("")
        return ("" as! Content, raw)
    }

    let raw = GeneratedContent(properties: [:])
    let content = try type.init(raw)
    return (content, raw)
}

private func toGeneratedContent(_ jsonString: String?) throws -> GeneratedContent {
    guard let jsonString, !jsonString.isEmpty else { return GeneratedContent(properties: [:]) }
    return try GeneratedContent(json: jsonString)
}

private func extractTextFromOutput(_ output: [JSONValue]?) -> String? {
    guard let output else { return nil }

    var textParts: [String] = []
    for block in output {
        if case let .object(obj) = block,
            case let .string(type)? = obj["type"],
            type == "message",
            case let .array(contentBlocks)? = obj["content"]
        {
            for contentBlock in contentBlocks {
                if case let .object(contentObj) = contentBlock,
                    case let .string(contentType)? = contentObj["type"],
                    contentType == "output_text",
                    case let .string(text)? = contentObj["text"]
                {
                    textParts.append(text)
                }
            }
        }
    }

    return textParts.isEmpty ? nil : textParts.joined()
}

private func extractJSONFromOutput(_ output: [JSONValue]?) -> String? {
    guard let output else { return nil }

    for block in output {
        if case let .object(obj) = block,
            case let .string(type)? = obj["type"],
            type == "message",
            case let .array(contentBlocks)? = obj["content"]
        {
            for contentBlock in contentBlocks {
                if case let .object(contentObj) = contentBlock,
                    case let .string(contentType)? = contentObj["type"],
                    contentType == "output_text",
                    case let .string(jsonString)? = contentObj["text"]
                {
                    return jsonString
                }
            }
        }
    }

    return nil
}

private func extractToolCallsFromOutput(_ output: [JSONValue]?) -> [OpenAIToolCall] {
    guard let output else { return [] }

    var toolCalls: [OpenAIToolCall] = []
    for block in output {
        if case let .object(obj) = block,
            case let .string(type)? = obj["type"]
        {
            // Handle direct function_call at top level
            if type == "function_call" {
                // Responses API uses "call_id", Chat Completions uses "id"
                guard
                    let id = (obj["call_id"] ?? obj["id"]).flatMap({
                        if case .string(let s) = $0 { return s } else { return nil }
                    }),
                    let name = obj["name"].flatMap({ if case .string(let s) = $0 { return s } else { return nil } })
                else { continue }

                let argsString: String?
                if let args = obj["arguments"] {
                    if case let .object(argObj) = args {
                        let argsData = try? JSONEncoder().encode(JSONValue.object(argObj))
                        argsString = argsData.flatMap { String(data: $0, encoding: .utf8) }
                    } else if case let .string(str) = args {
                        argsString = str
                    } else {
                        argsString = nil
                    }
                } else {
                    argsString = nil
                }

                let toolCall = OpenAIToolCall(
                    id: id,
                    type: "function",
                    function: OpenAIToolFunction(name: name, arguments: argsString)
                )
                toolCalls.append(toolCall)
            }
            // Handle message with nested content blocks
            else if type == "message", case let .array(contentBlocks)? = obj["content"] {
                for contentBlock in contentBlocks {
                    if case let .object(contentObj) = contentBlock,
                        case let .string(contentType)? = contentObj["type"],
                        (contentType == "tool_call" || contentType == "tool_use")
                    {
                        guard
                            let id = contentObj["id"].flatMap({
                                if case .string(let s) = $0 { return s } else { return nil }
                            }),
                            let name = contentObj["name"].flatMap({
                                if case .string(let s) = $0 { return s } else { return nil }
                            })
                        else { continue }

                        let argsString: String?
                        if let args = contentObj["arguments"] {
                            if case let .object(argObj) = args {
                                let argsData = try? JSONEncoder().encode(JSONValue.object(argObj))
                                argsString = argsData.flatMap { String(data: $0, encoding: .utf8) }
                            } else if case let .string(str) = args {
                                argsString = str
                            } else {
                                argsString = nil
                            }
                        } else if let input = contentObj["input"] {
                            if case let .object(argObj) = input {
                                let argsData = try? JSONEncoder().encode(JSONValue.object(argObj))
                                argsString = argsData.flatMap { String(data: $0, encoding: .utf8) }
                            } else if case let .string(str) = input {
                                argsString = str
                            } else {
                                argsString = nil
                            }
                        } else {
                            argsString = nil
                        }

                        let toolCall = OpenAIToolCall(
                            id: id,
                            type: "function",
                            function: OpenAIToolFunction(name: name, arguments: argsString)
                        )
                        toolCalls.append(toolCall)
                    }
                }
            }
        }
    }

    return toolCalls
}

// MARK: - Errors

enum OpenAILanguageModelError: LocalizedError {
    case noResponseGenerated

    var errorDescription: String? {
        switch self {
        case .noResponseGenerated:
            return "No response was generated by the model"
        }
    }
}

// MARK: - OpenAI Schema Helpers

private extension GenerationSchema {
    /// Converts this schema to a JSONValue with OpenAI strict mode requirements applied.
    ///
    /// OpenAI strict mode requires:
    /// 1. `additionalProperties: false` at the root
    /// 2. All properties (including optional ones) listed in `required`
    func toJSONValueForOpenAIStrictMode() throws -> JSONValue {
        let resolvedSchema = self.withResolvedRoot() ?? self

        let encoder = JSONEncoder()
        encoder.userInfo[GenerationSchema.omitAdditionalPropertiesKey] = false
        let schemaData = try encoder.encode(resolvedSchema)
        let jsonSchema = try JSONDecoder().decode(JSONSchema.self, from: schemaData)
        var jsonSchemaValue = try JSONValue(jsonSchema)

        if case .object(var schemaObj) = jsonSchemaValue {
            schemaObj["additionalProperties"] = .bool(false)

            if case .object(let properties)? = schemaObj["properties"],
                !properties.isEmpty
            {
                // OpenAI strict mode requires all properties to be listed as required,
                // even if the underlying schema marks them optional.
                let allPropertyNames = Array(properties.keys).sorted()
                schemaObj["required"] = .array(allPropertyNames.map { .string($0) })
            }

            jsonSchemaValue = .object(schemaObj)
        }

        return jsonSchemaValue
    }
}
