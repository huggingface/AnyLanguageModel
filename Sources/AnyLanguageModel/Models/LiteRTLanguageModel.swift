import Foundation

#if LiteRT
    @preconcurrency import LiteRTFoundation

    /// A language model that runs `.litertlm` models fully on-device via Google's
    /// [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime.
    ///
    /// Use this model to run Gemma 4 (and other LiteRT-LM models) on iOS and macOS
    /// with Metal GPU acceleration, including image understanding for models that
    /// ship a vision tower.
    ///
    /// ```swift
    /// let model = LiteRTLanguageModel(model: .gemma4_E2B)
    /// let session = LanguageModelSession(model: model)
    /// let response = try await session.respond(to: "What is the capital of France?")
    /// ```
    ///
    /// The model file is downloaded from Hugging Face on first use and cached under
    /// Application Support. Loading is lazy: the engine is brought up on the first
    /// request (or when ``prewarm(for:promptPrefix:)`` is called).
    ///
    /// Structured generation is prompt-driven (the JSON schema is included in the
    /// prompt and the response is parsed), and tool calling is supported for
    /// ``respond(within:to:generating:includeSchemaInPrompt:options:)``
    /// (not yet for streaming).
    public struct LiteRTLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available; loading errors surface when responding.
        public typealias UnavailableReason = Never

        private let engine: LazyEngine

        /// Creates a model from the swift-litert-lm catalog, downloading the
        /// `.litertlm` from Hugging Face on first use.
        ///
        /// - Parameters:
        ///   - model: The catalog model to run (for example, `.gemma4_E2B`).
        ///   - modalities: Towers to enable. Defaults to the model's default
        ///     modalities. Requesting an unsupported tower is ignored.
        ///   - storageDirectory: Where to keep the downloaded model. Defaults to
        ///     Application Support/LiteRTModels.
        ///   - allowUnsafeMemory: Bypass the device-RAM safety check.
        ///   - maxTokens: Context (KV cache) budget. Defaults to the catalog value.
        ///   - onDownloadProgress: Called during the first-run model download.
        public init(
            model: LiteRTModel,
            modalities: Modality? = nil,
            storageDirectory: URL? = nil,
            allowUnsafeMemory: Bool = false,
            maxTokens: Int? = nil,
            onDownloadProgress: (@Sendable (ModelDownloader.Progress) -> Void)? = nil
        ) {
            self.engine = LazyEngine {
                if !allowUnsafeMemory {
                    let ram = Int64(ProcessInfo.processInfo.physicalMemory)
                    if ram < model.minimumDeviceRAM {
                        throw LiteRTChatError.insufficientMemory(
                            haveBytes: ram,
                            needBytes: model.minimumDeviceRAM
                        )
                    }
                }
                let path = try await LiteRTChat.ensureModel(
                    model,
                    storageDirectory: storageDirectory,
                    onProgress: onDownloadProgress
                )
                var wanted = modalities ?? model.defaultModalities
                wanted.formIntersection(model.supportedModalities)
                return try await makeEngine(
                    modelPath: path,
                    modalities: wanted,
                    visionBackend: model.visionBackend,
                    audioBackend: model.audioBackend,
                    maxTokens: maxTokens ?? model.defaultMaxTokens,
                    visualTokenBudget: model.defaultVisualTokenBudget
                )
            }
        }

        /// Creates a model from a local `.litertlm` file. No download.
        ///
        /// - Parameters:
        ///   - modelFileURL: Absolute file URL of an on-disk `.litertlm`.
        ///   - modalities: Towers to bring up. Defaults to `.all`; only the ones
        ///     the model actually contains will work.
        ///   - visionBackend: Backend for the vision encoder. Defaults to `.cpu()`
        ///     (the safe choice for Gemma 4 on iOS).
        ///   - audioBackend: Backend for the audio encoder. Defaults to `.cpu()`.
        ///   - visualTokenBudget: Per-image visual-token cap (`nil` = engine default).
        ///   - maxTokens: Context (KV cache) budget.
        public init(
            modelFileURL: URL,
            modalities: Modality = .all,
            visionBackend: Backend = .cpu(),
            audioBackend: Backend = .cpu(),
            visualTokenBudget: Int32? = nil,
            maxTokens: Int = 2048
        ) {
            self.engine = LazyEngine {
                guard FileManager.default.fileExists(atPath: modelFileURL.path) else {
                    throw LiteRTChatError.modelFileNotFound(modelFileURL)
                }
                return try await makeEngine(
                    modelPath: modelFileURL.path,
                    modalities: modalities,
                    visionBackend: visionBackend,
                    audioBackend: audioBackend,
                    maxTokens: maxTokens,
                    visualTokenBudget: visualTokenBudget
                )
            }
        }

        /// Creates a model from any Hugging Face repo hosting a `.litertlm`,
        /// downloading it on first use.
        ///
        /// - Parameters:
        ///   - huggingFaceRepo: For example, `"litert-community/gemma-4-E4B-it-litert-lm"`.
        ///   - fileName: The `.litertlm` file in that repo.
        ///   - revision: Git revision or branch. Defaults to `main`.
        ///   - modalities: Defaults to text-only (`[]`) — the safe choice for an
        ///     unknown model. Pass `.textImage` or `.all` if the model ships those
        ///     encoders.
        ///   - visionBackend: Backend for the vision encoder. Defaults to `.cpu()`.
        ///   - audioBackend: Backend for the audio encoder. Defaults to `.cpu()`.
        ///   - visualTokenBudget: Per-image visual-token cap (`nil` = engine default).
        ///   - maxTokens: Context (KV cache) budget.
        ///   - storageDirectory: Where to keep the downloaded model.
        ///   - onDownloadProgress: Called during the first-run model download.
        public init(
            huggingFaceRepo: String,
            fileName: String,
            revision: String = "main",
            modalities: Modality = [],
            visionBackend: Backend = .cpu(),
            audioBackend: Backend = .cpu(),
            visualTokenBudget: Int32? = nil,
            maxTokens: Int = 2048,
            storageDirectory: URL? = nil,
            onDownloadProgress: (@Sendable (ModelDownloader.Progress) -> Void)? = nil
        ) {
            self.engine = LazyEngine {
                guard
                    let url = URL(
                        string:
                            "https://huggingface.co/\(huggingFaceRepo)/resolve/\(revision)/\(fileName)?download=true"
                    )
                else {
                    throw LiteRTChatError.modelFileNotFound(URL(fileURLWithPath: fileName))
                }
                let directory = try storageDirectory ?? LiteRTChat.defaultStorageDirectory()
                let destination = directory.appendingPathComponent(fileName)
                try await ModelDownloader.shared.download(
                    from: url,
                    to: destination,
                    expectedBytes: nil,
                    onProgress: onDownloadProgress
                )
                return try await makeEngine(
                    modelPath: destination.path,
                    modalities: modalities,
                    visionBackend: visionBackend,
                    audioBackend: audioBackend,
                    maxTokens: maxTokens,
                    visualTokenBudget: visualTokenBudget
                )
            }
        }

        public func prewarm(
            for session: LanguageModelSession,
            promptPrefix: Prompt?
        ) {
            let engine = self.engine
            Task { _ = try? await engine.ready() }
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let engine = try await self.engine.ready()

            let schemaJSON: String?
            if type == String.self {
                schemaJSON = nil
            } else {
                schemaJSON = try encodeSchema(type.generationSchema)
            }

            let tools = session.tools
            var plan = makePlan(
                from: session.transcript,
                fallbackPrompt: prompt.description,
                schemaJSON: includeSchemaInPrompt ? schemaJSON : nil,
                tools: tools
            )
            let sampler = makeSampler(for: options, structured: schemaJSON != nil || !tools.isEmpty)

            var entries: [Transcript.Entry] = []
            var text = ""
            var toolRounds = 0

            while true {
                let conversation = try await engine.createConversation(
                    with: ConversationConfig(
                        systemMessage: plan.systemMessage,
                        initialMessages: plan.history,
                        samplerConfig: sampler
                    )
                )

                text = ""
                for try await chunk in conversation.sendMessageStream(plan.prompt) {
                    text += chunk.toString
                }

                guard !tools.isEmpty,
                    toolRounds < maxToolRounds,
                    let parsed = parseToolCall(from: text, tools: tools)
                else { break }
                toolRounds += 1

                let resolution = try await resolveToolCall(
                    name: parsed.name,
                    argumentsJSON: parsed.arguments,
                    session: session
                )
                switch resolution {
                case .stop(let call):
                    entries.append(.toolCalls(Transcript.ToolCalls([call])))
                    return LanguageModelSession.Response(
                        content: "" as! Content,
                        rawContent: GeneratedContent(""),
                        transcriptEntries: ArraySlice(entries)
                    )
                case .invocation(let call, let output):
                    entries.append(.toolCalls(Transcript.ToolCalls([call])))
                    entries.append(.toolOutput(output))
                    plan = plan.continuing(afterModelText: text, toolOutput: output)
                }
            }

            if type == String.self {
                return LanguageModelSession.Response(
                    content: text as! Content,
                    rawContent: GeneratedContent(text),
                    transcriptEntries: ArraySlice(entries)
                )
            }

            let json = extractJSONObject(from: text) ?? text
            let generatedContent = try GeneratedContent(json: json)
            let content = try type.init(generatedContent)
            return LanguageModelSession.Response(
                content: content,
                rawContent: generatedContent,
                transcriptEntries: ArraySlice(entries)
            )
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            let lazyEngine = self.engine
            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            let engine = try await lazyEngine.ready()

                            let schemaJSON: String?
                            if type == String.self {
                                schemaJSON = nil
                            } else {
                                schemaJSON = try encodeSchema(type.generationSchema)
                            }

                            let plan = makePlan(
                                from: session.transcript,
                                fallbackPrompt: prompt.description,
                                schemaJSON: includeSchemaInPrompt ? schemaJSON : nil,
                                tools: []
                            )
                            let conversation = try await engine.createConversation(
                                with: ConversationConfig(
                                    systemMessage: plan.systemMessage,
                                    initialMessages: plan.history,
                                    samplerConfig: makeSampler(for: options, structured: schemaJSON != nil)
                                )
                            )

                            var text = ""
                            for try await chunk in conversation.sendMessageStream(plan.prompt) {
                                let delta = chunk.toString
                                guard !delta.isEmpty else { continue }
                                text += delta

                                if type == String.self {
                                    continuation.yield(
                                        .init(
                                            content: (text as! Content).asPartiallyGenerated(),
                                            rawContent: GeneratedContent(text)
                                        )
                                    )
                                } else if let json = extractJSONObject(from: text),
                                    let raw = try? GeneratedContent(json: json),
                                    let parsed = try? type.init(raw)
                                {
                                    continuation.yield(
                                        .init(
                                            content: parsed.asPartiallyGenerated(),
                                            rawContent: raw
                                        )
                                    )
                                } else {
                                    // Structured responses stream as incomplete JSON fragments.
                                    // Skip snapshots until the accumulated JSON parses cleanly.
                                }
                            }

                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }

                    continuation.onTermination = { _ in
                        task.cancel()
                    }
                }

            return LanguageModelSession.ResponseStream(stream: stream)
        }
    }

    // MARK: - Engine Bring-Up

    /// Brings up the engine on first use and shares it across requests.
    /// Engine bring-up loads multi-GB weights, so it must happen exactly once.
    private actor LazyEngine {
        private var task: Task<Engine, any Error>?
        private let bringUp: @Sendable () async throws -> Engine

        init(_ bringUp: @escaping @Sendable () async throws -> Engine) {
            self.bringUp = bringUp
        }

        func ready() async throws -> Engine {
            if task == nil {
                let bringUp = self.bringUp
                task = Task { try await bringUp() }
            }
            return try await task!.value
        }
    }

    private func makeEngine(
        modelPath: String,
        modalities: Modality,
        visionBackend: Backend,
        audioBackend: Backend,
        maxTokens: Int,
        visualTokenBudget: Int32?
    ) async throws -> Engine {
        ExperimentalFlags.optIntoExperimentalAPIs()
        if modalities.contains(.vision), let visualTokenBudget {
            ExperimentalFlags.visualTokenBudget = visualTokenBudget
        }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        let config = try EngineConfig(
            modelPath: modelPath,
            backend: .gpu,
            visionBackend: modalities.contains(.vision) ? visionBackend : nil,
            audioBackend: modalities.contains(.audio) ? audioBackend : nil,
            maxNumTokens: maxTokens,
            cacheDir: caches?.path,
            // The engine default is 1 image per conversation (a 2nd image
            // overwrites the 1st); allow several so multi-image chats work.
            maxNumImages: modalities.contains(.vision) ? 16 : nil
        )
        let engine = Engine(engineConfig: config)
        try await engine.initialize()
        return engine
    }

    // MARK: - Transcript → LiteRT Messages

    private struct GenerationPlan {
        var systemMessage: Message?
        var history: [Message]
        var prompt: Message

        /// Extends the plan after a tool round-trip: the trigger prompt and the
        /// model's tool-call text become history, and the tool result becomes the
        /// new trigger.
        func continuing(afterModelText text: String, toolOutput: Transcript.ToolOutput) -> GenerationPlan {
            var history = self.history
            history.append(prompt)
            history.append(Message(text, role: .model))
            let result = textContent(of: toolOutput.segments)
            let trigger = Message(
                "Tool \"\(toolOutput.toolName)\" returned: \(result)\nUse this result to answer the user.",
                role: .user
            )
            return GenerationPlan(systemMessage: systemMessage, history: history, prompt: trigger)
        }
    }

    /// Splits the session transcript into a system message, prior turns, and the
    /// message to generate from. The generation trigger is the last `.prompt` or
    /// (in a tool round-trip) the last `.toolOutput` entry.
    private func makePlan(
        from transcript: Transcript,
        fallbackPrompt: String,
        schemaJSON: String?,
        tools: [any Tool]
    ) -> GenerationPlan {
        let entries = Array(transcript)
        let triggerIndex = entries.lastIndex { entry in
            switch entry {
            case .prompt, .toolOutput: return true
            default: return false
            }
        }

        var systemText: [String] = []
        if !tools.isEmpty {
            systemText.append(toolInstructions(tools))
        }
        var history: [Message] = []
        var trigger: Message?

        for (index, entry) in entries.enumerated() {
            let isTrigger = (index == triggerIndex)
            switch entry {
            case .instructions(let instructions):
                systemText.append(textContent(of: instructions.segments))
            case .prompt(let prompt):
                var contents = messageContents(of: prompt.segments)
                if isTrigger, let schemaJSON, !schemaJSON.isEmpty {
                    contents.append(
                        .text(
                            "\n\nRespond with ONLY a JSON object that conforms to this JSON schema. "
                                + "Output valid JSON and nothing else:\n\(schemaJSON)"
                        )
                    )
                }
                let message = Message(contents: contents, role: .user)
                if isTrigger { trigger = message } else { history.append(message) }
            case .response(let response):
                history.append(Message(contents: [.text(textContent(of: response.segments))], role: .model))
            case .toolOutput(let output):
                let result = textContent(of: output.segments)
                let message = Message(
                    "Tool \"\(output.toolName)\" returned: \(result)\nUse this result to answer the user.",
                    role: .user
                )
                if isTrigger { trigger = message } else { history.append(message) }
            case .toolCalls:
                history.append(Message("[the assistant called a tool]", role: .model))
            }
        }

        let system = systemText.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return GenerationPlan(
            systemMessage: system.isEmpty ? nil : Message(system, role: .system),
            history: history,
            prompt: trigger ?? Message(fallbackPrompt, role: .user)
        )
    }

    /// Maps transcript segments to LiteRT content: text, structured content as
    /// JSON text, and images.
    private func messageContents(of segments: [Transcript.Segment]) -> [Content] {
        var contents: [Content] = []
        for segment in segments {
            switch segment {
            case .text(let text):
                if !text.content.isEmpty {
                    contents.append(.text(text.content))
                }
            case .structure(let structure):
                contents.append(.text(structure.content.jsonString))
            case .image(let image):
                switch image.source {
                case .data(let data, _):
                    contents.append(.imageData(data))
                case .url(let url):
                    if url.isFileURL {
                        contents.append(.imageFile(url.path))
                    } else if let data = try? Data(contentsOf: url) {
                        contents.append(.imageData(data))
                    }
                }
            }
        }
        return contents.isEmpty ? [.text("")] : contents
    }

    /// Concatenates the text of a segment list (non-text segments are ignored).
    private func textContent(of segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            if case .text(let text) = segment { return text.content } else { return nil }
        }.joined(separator: " ")
    }

    // MARK: - Structured Generation

    private func encodeSchema(_ schema: GenerationSchema) throws -> String {
        let resolvedSchema = schema.withResolvedRoot() ?? schema
        let data = try JSONEncoder().encode(resolvedSchema)
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// Extracts the first balanced JSON object from model text
    /// (strips surrounding prose and code fences).
    private func extractJSONObject(from text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let character = text[index]
            if inString {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    inString = false
                }
            } else if character == "\"" {
                inString = true
            } else if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 { return String(text[start ... index]) }
            }
            index = text.index(after: index)
        }
        return nil
    }

    // MARK: - Sampling

    private func makeSampler(for options: GenerationOptions, structured: Bool) -> SamplerConfig? {
        var topK = 40
        var topP = 0.95
        // Lower default temperature for structured / tool generation
        // (more reliable JSON).
        var temperature = structured ? 0.0 : 0.8
        if let explicit = options.temperature {
            temperature = explicit
        }
        if let sampling = options.sampling {
            switch sampling.mode {
            case .greedy:
                temperature = 0.0
            case .topK(let k, _):
                topK = k
            case .nucleus(let probabilityThreshold, _):
                topP = probabilityThreshold
            }
        }
        return try? SamplerConfig(topK: topK, topP: Float(topP), temperature: Float(temperature))
    }

    // MARK: - Tool Calling

    private let maxToolRounds = 4

    /// Describes the enabled tools and the tool-call JSON format for the prompt.
    private func toolInstructions(_ tools: [any Tool]) -> String {
        var lines = ["You can call tools to help answer the user. Available tools:"]
        for tool in tools {
            let parameters = (try? encodeSchema(tool.parameters)) ?? "{}"
            lines.append("- \(tool.name): \(tool.description). arguments schema: \(parameters)")
        }
        lines.append(
            "To call a tool, reply with ONLY this JSON and nothing else: "
                + "{\"tool_call\": {\"name\": \"<tool name>\", \"arguments\": { ... }}}. "
                + "If no tool is needed, answer the user directly."
        )
        return lines.joined(separator: "\n")
    }

    /// Parses a tool call from model output, if present and naming a known tool.
    private func parseToolCall(
        from text: String,
        tools: [any Tool]
    ) -> (name: String, arguments: String)? {
        guard let json = extractJSONObject(from: text),
            let data = json.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let call = object["tool_call"] as? [String: Any],
            let name = call["name"] as? String,
            tools.contains(where: { $0.name == name })
        else { return nil }
        let arguments = call["arguments"] ?? [String: Any]()
        let argumentsData = (try? JSONSerialization.data(withJSONObject: arguments)) ?? Data("{}".utf8)
        return (name, String(data: argumentsData, encoding: .utf8) ?? "{}")
    }

    private enum ToolResolution {
        case stop(call: Transcript.ToolCall)
        case invocation(call: Transcript.ToolCall, output: Transcript.ToolOutput)
    }

    private func resolveToolCall(
        name: String,
        argumentsJSON: String,
        session: LanguageModelSession
    ) async throws -> ToolResolution {
        let arguments = (try? GeneratedContent(json: argumentsJSON)) ?? GeneratedContent(properties: [:])
        let call = Transcript.ToolCall(id: UUID().uuidString, toolName: name, arguments: arguments)

        if let delegate = session.toolExecutionDelegate {
            await delegate.didGenerateToolCalls([call], in: session)
        }

        var decision: ToolExecutionDecision = .execute
        if let delegate = session.toolExecutionDelegate {
            decision = await delegate.toolCallDecision(for: call, in: session)
        }

        switch decision {
        case .stop:
            return .stop(call: call)
        case .provideOutput(let segments):
            let output = Transcript.ToolOutput(id: call.id, toolName: call.toolName, segments: segments)
            if let delegate = session.toolExecutionDelegate {
                await delegate.didExecuteToolCall(call, output: output, in: session)
            }
            return .invocation(call: call, output: output)
        case .execute:
            guard let tool = session.tools.first(where: { $0.name == name }) else {
                let message = Transcript.Segment.text(.init(content: "Tool not found: \(name)"))
                let output = Transcript.ToolOutput(id: call.id, toolName: name, segments: [message])
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                return .invocation(call: call, output: output)
            }

            do {
                let segments = try await tool.makeOutputSegments(from: call.arguments)
                let output = Transcript.ToolOutput(id: call.id, toolName: tool.name, segments: segments)
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                return .invocation(call: call, output: output)
            } catch {
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didFailToolCall(call, error: error, in: session)
                }
                throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
            }
        }
    }
#endif
