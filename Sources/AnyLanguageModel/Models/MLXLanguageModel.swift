import Foundation

#if canImport(UIKit) && canImport(CoreImage)
    import UIKit
    import CoreImage
#endif

#if canImport(AppKit) && canImport(CoreImage)
    import AppKit
    import CoreImage
#endif

#if MLX
    import JSONSchema
    import MLXLMCommon
    import MLX
    import MLXVLM
    import Tokenizers
    import Hub

    /// Wrapper to store model availability state in NSCache.
    private final class CachedModelState: NSObject, @unchecked Sendable {
        enum Value {
            case loaded(ModelContext)
            case failed(String)
        }

        let value: Value

        init(_ value: Value) {
            self.value = value
        }
    }

    /// Coordinates a bounded in-memory cache with structured, coalesced loading.
    private final class ModelContextCache {
        private let cache: NSCache<NSString, CachedModelState>
        private let inFlight = Locked<[String: Task<CachedModelState, Error>]>([:])

        /// Creates a cache with a count-based eviction limit.
        init(countLimit: Int) {
            let cache = NSCache<NSString, CachedModelState>()
            cache.countLimit = countLimit
            self.cache = cache
        }

        /// Returns a cached context or loads it exactly once per key.
        func context(
            for key: String,
            loader: @escaping @Sendable () async throws -> ModelContext
        ) async throws -> ModelContext {
            let cacheKey = key as NSString
            if let cached = cache.object(forKey: cacheKey),
                case .loaded(let context) = cached.value
            {
                return context
            }

            if let task = inFlightTask(for: key) {
                let cached = try await task.value
                if case .loaded(let context) = cached.value {
                    return context
                }
                throw CancellationError()
            }

            let task = Task {
                let context = try await loader()
                return CachedModelState(.loaded(context))
            }
            setInFlight(task, for: key)

            do {
                let cached = try await task.value
                cache.setObject(cached, forKey: cacheKey)
                clearInFlight(for: key)
                if case .loaded(let context) = cached.value {
                    return context
                }
                throw CancellationError()
            } catch {
                // Don't treat cancellations as load failures.
                if error is CancellationError || Task.isCancelled {
                    cache.removeObject(forKey: cacheKey)
                    clearInFlight(for: key)
                    throw error
                }
                cache.setObject(
                    CachedModelState(.failed(String(reflecting: error))),
                    forKey: cacheKey
                )
                clearInFlight(for: key)
                throw error
            }
        }

        /// Removes a cached context for the key.
        func remove(for key: String) {
            cache.removeObject(forKey: key as NSString)
        }

        /// Clears all cached contexts.
        func removeAll() {
            cache.removeAllObjects()
        }

        /// Returns whether a cached context exists for the key.
        func contains(_ key: String) -> Bool {
            guard let cached = cache.object(forKey: key as NSString) else {
                return false
            }
            if case .loaded = cached.value {
                return true
            }
            return false
        }

        /// Returns a description of the most recent load failure for the key.
        func failureDescription(for key: String) -> String? {
            guard let cached = cache.object(forKey: key as NSString) else {
                return nil
            }
            if case .failed(let description) = cached.value {
                return description
            }
            return nil
        }

        /// Cancels in-flight work and removes cached data for the key.
        func removeAndCancel(for key: String) async {
            let task = removeInFlight(for: key)
            task?.cancel()
            cache.removeObject(forKey: key as NSString)
        }

        /// Cancels all in-flight work and clears cached data.
        func removeAllAndCancel() async {
            let tasks = removeAllInFlight()
            tasks.forEach { $0.cancel() }
            cache.removeAllObjects()
        }

        private func inFlightTask(for key: String) -> Task<CachedModelState, Error>? {
            inFlight.withLock { $0[key] }
        }

        private func setInFlight(_ task: Task<CachedModelState, Error>, for key: String) {
            inFlight.withLock { $0[key] = task }
        }

        private func clearInFlight(for key: String) {
            inFlight.withLock { $0[key] = nil }
        }

        private func removeInFlight(for key: String) -> Task<CachedModelState, Error>? {
            inFlight.withLock {
                let task = $0[key]
                $0[key] = nil
                return task
            }
        }

        private func removeAllInFlight() -> [Task<CachedModelState, Error>] {
            inFlight.withLock {
                let tasks = Array($0.values)
                $0.removeAll()
                return tasks
            }
        }
    }

    /// Shared cache across MLXLanguageModel instances.
    private nonisolated(unsafe) let modelCache = ModelContextCache(countLimit: 3)

    // MARK: - MLXLanguageModel

    /// A language model that runs locally using MLX.
    ///
    /// Use this model to run language models on Apple silicon using the MLX framework.
    /// Models are automatically downloaded and cached when first used.
    ///
    /// ```swift
    /// let model = MLXLanguageModel(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit")
    /// ```
    public struct MLXLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        public enum UnavailableReason: Sendable, Equatable, Hashable {
            /// The model has not been loaded into memory yet.
            case notLoaded
            /// The model failed to load and includes the underlying error details.
            case failedToLoad(String)
        }

        /// The model identifier.
        public let modelId: String

        /// The Hub API instance for downloading models.
        public let hub: HubApi?

        /// The local directory containing the model files.
        public let directory: URL?

        /// Creates an MLX language model.
        ///
        /// - Parameters:
        ///   - modelId: The model identifier (for example, "mlx-community/Llama-3.2-3B-Instruct-4bit").
        ///   - hub: An optional Hub API instance for downloading models. If not provided, the default Hub API is used.
        ///   - directory: An optional local directory URL containing the model files. If provided, the model is loaded from this directory instead of downloading.
        public init(modelId: String, hub: HubApi? = nil, directory: URL? = nil) {
            self.modelId = modelId
            self.hub = hub
            self.directory = directory
        }

        /// The current availability of this model in memory.
        public var availability: Availability<UnavailableReason> {
            let key = directory?.absoluteString ?? modelId
            if modelCache.contains(key) {
                return .available
            }

            if let failureDescription = modelCache.failureDescription(for: key) {
                return .unavailable(.failedToLoad(failureDescription))
            }

            return .unavailable(.notLoaded)
        }

        /// Removes this model from the shared cache and cancels any in-flight load.
        ///
        /// Call this to free memory when the model is no longer needed.
        /// The model will be reloaded automatically on the next request.
        public func removeFromCache() async {
            let key = directory?.absoluteString ?? modelId
            await modelCache.removeAndCancel(for: key)
        }

        /// Removes all MLX models from the shared cache and cancels in-flight loads.
        public static func removeAllFromCache() async {
            await modelCache.removeAllAndCancel()
        }

        /// Get or load model context with caching
        private func loadContext(modelId: String, hub: HubApi?, directory: URL?) async throws -> ModelContext {
            let key = directory?.absoluteString ?? modelId

            return try await modelCache.context(for: key) {
                if let directory {
                    return try await loadModel(directory: directory)
                }

                return try await loadModel(hub: hub ?? HubApi(), id: modelId)
            }
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            // Get cached or load fresh ModelContext
            let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)

            if type != String.self {
                let jsonString = try await generateStructuredJSON(
                    context: context,
                    session: session,
                    prompt: prompt,
                    schema: type.generationSchema,
                    options: options,
                    includeSchemaInPrompt: includeSchemaInPrompt
                )
                let generatedContent = try GeneratedContent(json: jsonString)
                let content = try type.init(generatedContent)
                return LanguageModelSession.Response(
                    content: content,
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice([])
                )
            }

            // Convert session tools to MLX ToolSpec format
            let toolSpecs: [ToolSpec]? =
                session.tools.isEmpty
                ? nil
                : session.tools.map { tool in
                    convertToolToMLXSpec(tool)
                }

            // Map AnyLanguageModel GenerationOptions to MLX GenerateParameters
            let generateParameters = toGenerateParameters(options)

            // Build chat history from full transcript
            var chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

            var allTextChunks: [String] = []
            var allEntries: [Transcript.Entry] = []

            // Loop until no more tool calls
            while true {
                // Build user input with current chat history and tools
                let userInput = MLXLMCommon.UserInput(
                    chat: chat,
                    processing: .init(resize: .init(width: 512, height: 512)),
                    tools: toolSpecs,
                )
                let lmInput = try await context.processor.prepare(input: userInput)

                // Generate
                let stream = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: generateParameters,
                    context: context
                )

                var chunks: [String] = []
                var collectedToolCalls: [MLXLMCommon.ToolCall] = []

                for await item in stream {
                    switch item {
                    case .chunk(let text):
                        chunks.append(text)
                    case .info:
                        break
                    case .toolCall(let call):
                        collectedToolCalls.append(call)
                    }
                }

                let assistantText = chunks.joined()
                allTextChunks.append(assistantText)

                // Add assistant response to chat history
                if !assistantText.isEmpty {
                    chat.append(.assistant(assistantText))
                }

                // If there are tool calls, execute them and continue
                if !collectedToolCalls.isEmpty {
                    let resolution = try await resolveToolCalls(collectedToolCalls, session: session)
                    switch resolution {
                    case .stop(let calls):
                        if !calls.isEmpty {
                            allEntries.append(.toolCalls(Transcript.ToolCalls(calls)))
                        }
                        return LanguageModelSession.Response(
                            content: "" as! Content,
                            rawContent: GeneratedContent(""),
                            transcriptEntries: ArraySlice(allEntries)
                        )
                    case .invocations(let invocations):
                        if !invocations.isEmpty {
                            allEntries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))

                            // Execute each tool and add results to chat
                            for invocation in invocations {
                                allEntries.append(.toolOutput(invocation.output))

                                // Convert tool output to JSON string for MLX
                                let toolResultJSON = toolOutputToJSON(invocation.output)
                                chat.append(.tool(toolResultJSON))
                            }

                            // Continue loop to generate with tool results
                            continue
                        }
                    }
                }

                // No more tool calls, exit loop
                break
            }

            let text = allTextChunks.joined()
            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(allEntries)
            )
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            guard type == String.self else {
                fatalError("MLXLanguageModel streaming only supports String content")
            }

            let modelId = self.modelId
            let hub = self.hub
            let directory = self.directory

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                let task = Task { @Sendable in
                    do {
                        // Get cached or load fresh ModelContext
                        let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)

                        // Build chat inside task to avoid Sendable issues
                        let generateParameters = toGenerateParameters(options)
                        let chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

                        let userInput = MLXLMCommon.UserInput(
                            chat: chat,
                            processing: .init(resize: .init(width: 512, height: 512)),
                            tools: nil
                        )
                        let lmInput = try await context.processor.prepare(input: userInput)

                        let mlxStream = try MLXLMCommon.generate(
                            input: lmInput,
                            parameters: generateParameters,
                            context: context
                        )

                        var accumulatedText = ""
                        for await item in mlxStream {
                            if Task.isCancelled { break }

                            switch item {
                            case .chunk(let text):
                                accumulatedText += text
                                let raw = GeneratedContent(accumulatedText)
                                let content: Content.PartiallyGenerated = (accumulatedText as! Content)
                                    .asPartiallyGenerated()
                                continuation.yield(.init(content: content, rawContent: raw))
                            case .info, .toolCall:
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

        /// Prewarms the model
        public func prewarm(
            for session: LanguageModelSession,
            promptPrefix: Prompt?
        ) {
            let modelId = self.modelId
            let hub = self.hub
            let directory = self.directory

            Task {
                do {
                    _ = try await loadContext(modelId: modelId, hub: hub, directory: directory)
                } catch {
                    // Ignore errors during prewarm
                }
            }
        }
    }

    // MARK: - Options Mapping

    private func toGenerateParameters(_ options: GenerationOptions) -> MLXLMCommon.GenerateParameters {
        MLXLMCommon.GenerateParameters(
            maxTokens: options.maximumResponseTokens,
            maxKVSize: nil,
            kvBits: nil,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: Float(options.temperature ?? 0.6),
            topP: 1.0,
            repetitionPenalty: nil,
            repetitionContextSize: 20
        )
    }

    /// Builds MLX parameters tuned for structured generation.
    private func toStructuredGenerateParameters(_ options: GenerationOptions) -> MLXLMCommon.GenerateParameters {
        MLXLMCommon.GenerateParameters(
            maxTokens: options.maximumResponseTokens,
            maxKVSize: nil,
            kvBits: nil,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: Float(options.temperature ?? 0.2),
            topP: 0.95,
            repetitionPenalty: 1.1,
            repetitionContextSize: 64
        )
    }

    // MARK: - Transcript Conversion

    private func convertTranscriptToMLXChat(
        session: LanguageModelSession,
        fallbackPrompt: String
    ) -> [MLXLMCommon.Chat.Message] {
        var chat: [MLXLMCommon.Chat.Message] = []

        // Check if instructions are already in transcript
        let hasInstructionsInTranscript = session.transcript.contains {
            if case .instructions = $0 { return true }
            return false
        }

        // Add instructions from session if present and not in transcript
        if !hasInstructionsInTranscript,
            let instructions = session.instructions?.description,
            !instructions.isEmpty
        {
            chat.append(.init(role: .system, content: instructions))
        }

        // Convert each transcript entry
        for entry in session.transcript {
            switch entry {
            case .instructions(let instr):
                chat.append(makeMLXChatMessage(from: instr.segments, role: .system))

            case .prompt(let prompt):
                chat.append(makeMLXChatMessage(from: prompt.segments, role: .user))

            case .response(let response):
                let content = response.segments.map { extractText(from: $0) }.joined(separator: "\n")
                chat.append(.assistant(content))

            case .toolCalls:
                // Tool calls are handled inline during generation loop
                break

            case .toolOutput(let toolOutput):
                let content = toolOutput.segments.map { extractText(from: $0) }.joined(separator: "\n")
                chat.append(.tool(content))
            }
        }

        // If no user message in transcript, add fallback prompt
        let hasUserMessage = chat.contains { $0.role == .user }
        if !hasUserMessage {
            chat.append(.init(role: .user, content: fallbackPrompt))
        }

        return chat
    }

    private func extractText(from segment: Transcript.Segment) -> String {
        switch segment {
        case .text(let text):
            return text.content
        case .structure(let structured):
            return structured.content.jsonString
        case .image:
            return ""
        }
    }

    private func makeMLXChatMessage(
        from segments: [Transcript.Segment],
        role: MLXLMCommon.Chat.Message.Role
    ) -> MLXLMCommon.Chat.Message {
        var textParts: [String] = []
        var images: [MLXLMCommon.UserInput.Image] = []

        for segment in segments {
            switch segment {
            case .image(let imageSegment):
                switch imageSegment.source {
                case .url(let url):
                    images.append(.url(url))
                case .data(let data, _):
                    #if canImport(UIKit)
                        if let uiImage = UIKit.UIImage(data: data),
                            let ciImage = CIImage(image: uiImage)
                        {
                            images.append(.ciImage(ciImage))
                        }
                    #elseif canImport(AppKit)
                        if let nsImage = AppKit.NSImage(data: data),
                            let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
                        {
                            let ciImage = CIImage(cgImage: cgImage)
                            images.append(.ciImage(ciImage))
                        }
                    #endif
                }
            default:
                let text = extractText(from: segment)
                if !text.isEmpty {
                    textParts.append(text)
                }
            }
        }

        let content = textParts.joined(separator: "\n")
        return MLXLMCommon.Chat.Message(role: role, content: content, images: images)
    }

    // MARK: - Tool Conversion

    private func convertToolToMLXSpec(_ tool: any Tool) -> ToolSpec {
        // Convert AnyLanguageModel's GenerationSchema to JSON-compatible dictionary
        let parametersDict: [String: any Sendable]
        do {
            let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
            let encoder = JSONEncoder()
            let data = try encoder.encode(resolvedSchema)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                parametersDict = try convertToSendableJSONObject(json)
            } else {
                parametersDict = makeEmptyJSONSchemaObject()
            }
        } catch {
            parametersDict = makeEmptyJSONSchemaObject()
        }

        let functionSpec: [String: any Sendable] = [
            "name": tool.name,
            "description": tool.description,
            "parameters": parametersDict,
        ]

        let toolSpec: ToolSpec = [
            "type": "function",
            "function": functionSpec,
        ]

        return toolSpec
    }

    private func makeEmptyJSONSchemaObject() -> [String: any Sendable] {
        [
            "type": "object",
            "properties": [String: any Sendable](),
            "required": [String](),
        ]
    }

    private func convertToSendableJSONObject(_ object: [String: Any]) throws -> [String: any Sendable] {
        var converted: [String: any Sendable] = [:]
        converted.reserveCapacity(object.count)

        for (key, value) in object {
            converted[key] = try convertToSendableJSONValue(value)
        }
        return converted
    }

    private func convertToSendableJSONValue(_ value: Any) throws -> any Sendable {
        if value is NSNull { return MLXLMCommon.JSONValue.null }
        if let stringValue = value as? String { return stringValue }
        if let boolValue = value as? Bool { return boolValue }
        if let intValue = value as? Int { return intValue }
        if let doubleValue = value as? Double { return doubleValue }
        if let numberValue = value as? NSNumber {
            return numberValue.doubleValue
        }
        if let arrayValue = value as? [Any] {
            return try arrayValue.map { try convertToSendableJSONValue($0) }
        }
        if let dictionaryValue = value as? [String: Any] {
            return try convertToSendableJSONObject(dictionaryValue)
        }

        throw MLXLanguageModelError.unsupportedJSONValueType
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

    private func resolveToolCalls(
        _ toolCalls: [MLXLMCommon.ToolCall],
        session: LanguageModelSession
    ) async throws -> ToolResolutionOutcome {
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
            let args = try toGeneratedContent(call.function.arguments)
            let callID = UUID().uuidString
            transcriptCalls.append(
                Transcript.ToolCall(
                    id: callID,
                    toolName: call.function.name,
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

    private func toGeneratedContent(_ args: [String: MLXLMCommon.JSONValue]) throws -> GeneratedContent {
        let data = try JSONEncoder().encode(args)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        return try GeneratedContent(json: json)
    }

    private func toolOutputToJSON(_ output: Transcript.ToolOutput) -> String {
        // Extract text content from segments
        var textParts: [String] = []
        for segment in output.segments {
            switch segment {
            case .text(let textSegment):
                textParts.append(textSegment.content)
            case .structure(let structuredSegment):
                // structured content already has jsonString property
                textParts.append(structuredSegment.content.jsonString)
            case .image:
                // Image segments are not supported in MLX tool output
                break
            }
        }
        return textParts.joined(separator: "\n")
    }

    /// Builds a JSONSchema-informed prompt for structured output.
    private func schemaPrompt(for schema: GenerationSchema) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard
            let data = try? encoder.encode(schema),
            let jsonSchema = try? JSONDecoder().decode(JSONSchema.self, from: data),
            let schemaJSON = String(data: data, encoding: .utf8)
        else {
            return schema.schemaPrompt()
        }

        var header = "Respond with valid JSON matching this \(jsonSchema.typeName) schema"
        if let description = jsonSchema.description, !description.isEmpty {
            header += " (\(description))"
        }

        if let constValue = jsonSchema.const,
            let data = try? encoder.encode(constValue),
            let constString = String(data: data, encoding: .utf8)
        {
            header += ". Expected value: \(constString)"
        } else if let enumValues = jsonSchema.enum, !enumValues.isEmpty,
            let data = try? encoder.encode(JSONValue.array(enumValues)),
            let enumString = String(data: data, encoding: .utf8)
        {
            header += ". Allowed values: \(enumString)"
        }

        return "\(header):\n\(schemaJSON)"
    }

    // MARK: - Structured JSON Generation

    /// Errors that can occur when using MLXLanguageModel.
    public enum MLXLanguageModelError: Error, LocalizedError {
        case invalidVocabSize
        case unsupportedJSONValueType

        public var errorDescription: String? {
            switch self {
            case .invalidVocabSize:
                return "Invalid vocabulary size for model output"
            case .unsupportedJSONValueType:
                return "Unsupported JSON value type for schema conversion"
            }
        }
    }

    private func generateStructuredJSON(
        context: ModelContext,
        session: LanguageModelSession,
        prompt: Prompt,
        schema: GenerationSchema,
        options: GenerationOptions,
        includeSchemaInPrompt: Bool
    ) async throws -> String {
        let maxTokens = options.maximumResponseTokens ?? 512
        let generateParameters = toStructuredGenerateParameters(options)

        let baseChat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)
        let schemaPrompt = includeSchemaInPrompt ? schemaPrompt(for: schema) : nil
        let chat = normalizeChatForStructuredGeneration(baseChat, schemaPrompt: schemaPrompt)
        let userInput = MLXLMCommon.UserInput(
            chat: chat,
            processing: .init(resize: .init(width: 512, height: 512)),
            tools: nil
        )
        let lmInput = try await context.processor.prepare(input: userInput)

        let backend = try MLXTokenBackend(
            context: context,
            input: lmInput,
            parameters: generateParameters,
            maximumTokens: maxTokens,
            endTokens: []
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let json = try await generator.generate()
        // Ensure pending MLX operations complete before returning JSON.
        // This synchronization can be a performance cost if called frequently.
        Stream().synchronize()
        return json
    }

    /// Merges system prompts and schema instructions into a user message.
    /// Consecutive messages with the same role are merged to reduce prompt tokens.
    private func normalizeChatForStructuredGeneration(
        _ chat: [MLXLMCommon.Chat.Message],
        schemaPrompt: String?
    ) -> [MLXLMCommon.Chat.Message] {
        guard let schemaPrompt, !schemaPrompt.isEmpty else {
            return chat
        }

        var systemMessageParts: [String] = []
        systemMessageParts.append(schemaPrompt)

        var messages: [MLXLMCommon.Chat.Message] = []
        messages.reserveCapacity(chat.count)

        for message in chat {
            if message.role == .system {
                systemMessageParts.append(message.content)
                continue
            }

            if let last = messages.last, last.role == message.role {
                let merged = MLXLMCommon.Chat.Message(role: last.role, content: "\(last.content)\n\(message.content)")
                messages.removeLast()
                messages.append(merged)
            } else {
                messages.append(message)
            }
        }

        let systemPrefix =
            systemMessageParts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")

        guard !systemPrefix.isEmpty else {
            return messages
        }

        if let firstUserIndex = messages.firstIndex(where: { $0.role == .user }) {
            let existing = messages[firstUserIndex].content
            messages[firstUserIndex] = MLXLMCommon.Chat.Message(role: .user, content: "\(systemPrefix)\n\n\(existing)")
            return messages
        }

        messages.insert(.init(role: .user, content: systemPrefix), at: 0)
        return messages
    }

    private struct MLXTokenBackend: TokenBackend {
        let model: any MLXLMCommon.LanguageModel
        let tokenizer: any Tokenizer
        var state: MLXLMCommon.LMOutput.State?
        var cache: [MLXLMCommon.KVCache]
        var processor: MLXLMCommon.LogitProcessor?
        let sampler: MLXLMCommon.LogitSampler
        let tokensExcludedFromRepetitionPenalty: Set<Int>
        let endTokens: Set<Int>

        var currentLogits: MLXArray
        let vocabSize: Int
        let eosToken: Int
        var remainingTokens: Int
        let totalTokenBudget: Int

        init(
            context: ModelContext,
            input: MLXLMCommon.LMInput,
            parameters: MLXLMCommon.GenerateParameters,
            maximumTokens: Int,
            endTokens: Set<Int>? = nil
        ) throws {
            self.model = context.model
            self.tokenizer = context.tokenizer
            self.state = nil
            self.cache = context.model.newCache(parameters: parameters)
            self.processor = parameters.processor()
            self.sampler = parameters.sampler()
            self.remainingTokens = maximumTokens
            self.totalTokenBudget = maximumTokens
            guard let eosTokenId = context.tokenizer.eosTokenId else {
                throw MLXLanguageModelError.invalidVocabSize
            }
            self.eosToken = eosTokenId
            if let endTokens {
                self.endTokens = endTokens
            } else {
                self.endTokens = Self.buildEndTokens(
                    eosTokenId: eosTokenId,
                    tokenizer: context.tokenizer,
                    configuration: context.configuration
                )
            }

            self.tokensExcludedFromRepetitionPenalty = Self.buildTokensExcludedFromRepetitionPenalty(
                tokenizer: context.tokenizer
            )

            processor?.prompt(input.text.tokens)

            let prepareResult = try context.model.prepare(
                input,
                cache: cache,
                windowSize: parameters.prefillStepSize
            )

            let output: MLXLMCommon.LMOutput
            switch prepareResult {
            case .tokens(let tokensToProcess):
                output = context.model(
                    tokensToProcess[text: .newAxis],
                    cache: cache,
                    state: state
                )
            case .logits(let logitsOutput):
                output = logitsOutput
            }

            self.state = output.state
            self.currentLogits = output.logits

            guard output.logits.shape.count >= 1 else {
                throw MLXLanguageModelError.invalidVocabSize
            }
            self.vocabSize = output.logits.shape.last ?? 0
            guard self.vocabSize > 0 else {
                throw MLXLanguageModelError.invalidVocabSize
            }
        }

        private static func buildEndTokens(
            eosTokenId: Int,
            tokenizer: any Tokenizer,
            configuration: ModelConfiguration
        ) -> Set<Int> {
            var tokens: Set<Int> = [eosTokenId]

            // If the tokenizer declares an EOS token string, prefer treating its ID as an end token too.
            // Some chat models use a string EOS marker (e.g. "<end_of_turn>") whose ID may differ from eosTokenId.
            if let eosString = tokenizer.eosToken, let eosStringId = tokenizer.convertTokenToId(eosString) {
                tokens.insert(eosStringId)
            }

            for tokenString in configuration.extraEOSTokens {
                if let id = tokenizer.convertTokenToId(tokenString) {
                    tokens.insert(id)
                }
            }
            return tokens
        }

        func isSpecialToken(_ token: Int) -> Bool {
            // Use swift-transformers' own special token registry (skipSpecialTokens) instead of guessing.
            let raw = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
            guard !raw.isEmpty else { return false }
            let filtered = tokenizer.decode(tokens: [token], skipSpecialTokens: true)
            return filtered.isEmpty
        }

        private static func buildTokensExcludedFromRepetitionPenalty(tokenizer: any Tokenizer) -> Set<Int> {
            let excludedTexts = ["{", "}", "[", "]", ",", ":", "\""]
            var excluded = Set<Int>()
            excluded.reserveCapacity(excludedTexts.count * 2)

            for text in excludedTexts {
                let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
                for token in tokens {
                    excluded.insert(token)
                }
            }

            return excluded
        }

        func tokenize(_ text: String) throws -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: false)
        }

        func tokenText(_ token: Int) -> String? {
            let decoded = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
            return decoded.isEmpty ? nil : decoded
        }

        mutating func decode(_ token: Int) async throws {
            let inputText = MLXLMCommon.LMInput.Text(tokens: MLXArray([Int32(token)]))
            let output = model(
                inputText[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: state
            )
            state = output.state
            currentLogits = output.logits
            remainingTokens -= 1

            if !tokensExcludedFromRepetitionPenalty.contains(token) {
                let tokenArray = MLXArray(Int32(token))
                processor?.didSample(token: tokenArray)
            }
        }

        mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
            guard !allowedTokens.isEmpty else {
                throw ConstrainedGenerationError.tokenizationFailed
            }

            var logits = currentLogits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            if logits.dtype == .bfloat16 {
                logits = logits.asType(.float32)
            }

            let allowedIndices = MLXArray(allowedTokens.map { UInt32($0) })
            let maskedLogits = full(logits.shape, values: -Float.infinity)
            maskedLogits[0..., allowedIndices] = logits[0..., allowedIndices]

            let sampledToken = sampler.sample(logits: maskedLogits)
            return sampledToken.item(Int.self)
        }
    }
#endif  // MLX
