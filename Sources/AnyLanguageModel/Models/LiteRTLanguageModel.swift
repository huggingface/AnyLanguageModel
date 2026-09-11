import Foundation

#if LiteRT && (os(iOS) || os(macOS)) && !targetEnvironment(macCatalyst)
    @preconcurrency import LiteRTLM
    import class HuggingFace.HubClient
    import enum HuggingFace.Repo

    /// A language model that runs `.litertlm` models fully on-device
    /// via Google's [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime.
    ///
    /// Use this model to run Gemma 4 (and other LiteRT-LM models)
    /// on iOS and macOS with Metal GPU acceleration,
    /// including image understanding for models that ship a vision tower.
    ///
    /// ```swift
    /// let model = LiteRTLanguageModel(modelFileURL: modelURL)
    /// let session = LanguageModelSession(model: model)
    /// let response = try await session.respond(to: "What is the capital of France?")
    /// ```
    ///
    /// Hugging Face models are downloaded on first use and stored in the Hub cache.
    /// Loading is lazy:
    /// the engine is brought up on the first request
    /// (or when ``prewarm(for:promptPrefix:)`` is called).
    ///
    /// Structured generation is prompt-driven
    /// (the JSON schema is included in the prompt and the response is parsed),
    /// and tool calling is supported for
    /// ``respond(within:to:generating:includeSchemaInPrompt:options:)``
    /// (not yet for streaming).
    public struct LiteRTLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available;
        /// loading errors surface when responding.
        public typealias UnavailableReason = Never

        private let engine: LiteRTModelLoader
        private let imageSession: URLSession

        /// Creates a model from a local `.litertlm` file.
        ///
        /// - Parameters:
        ///   - modelFileURL: File URL of an on-disk `.litertlm` model.
        ///   - backend: Backend for text generation.
        ///     Defaults to Metal GPU.
        ///   - visionBackend: Backend for the vision encoder.
        ///     Pass `.cpu()` for a model with image support;
        ///     `nil` disables vision.
        ///   - audioBackend: Backend for the audio encoder;
        ///     `nil` disables audio.
        ///   - maxTokens: Context (KV cache) budget.
        public init(
            modelFileURL: URL,
            backend: Backend = .gpu,
            visionBackend: Backend? = nil,
            audioBackend: Backend? = nil,
            maxTokens: Int = 2048
        ) {
            self.imageSession = .shared
            self.engine = LiteRTModelLoader {
                guard modelFileURL.isFileURL,
                    FileManager.default.fileExists(atPath: modelFileURL.path)
                else {
                    throw CocoaError(.fileReadNoSuchFile, userInfo: [NSURLErrorKey: modelFileURL])
                }
                return try await makeEngine(
                    modelPath: modelFileURL.path,
                    backend: backend,
                    visionBackend: visionBackend,
                    audioBackend: audioBackend,
                    maxTokens: maxTokens
                )
            }
        }

        /// Creates a model from a Hugging Face repository hosting a `.litertlm` file.
        /// The file is downloaded lazily using the Hub client's cache and authentication.
        ///
        /// - Parameters:
        ///   - huggingFaceRepo: The Hugging Face model repository identifier.
        ///   - fileName: The path to the `.litertlm` file within the repository.
        ///   - revision: Git revision or branch.
        ///     Defaults to `main`.
        ///   - backend: Backend for text generation.
        ///     Defaults to Metal GPU.
        ///   - visionBackend: Backend for the vision encoder;
        ///     `nil` disables vision.
        ///   - audioBackend: Backend for the audio encoder;
        ///     `nil` disables audio.
        ///   - maxTokens: Context (KV cache) budget.
        ///   - hub: Optional Hub client for authentication and cache configuration.
        ///   - downloadProgress: Optional progress object for the model download.
        public init(
            huggingFaceRepo: String,
            fileName: String,
            revision: String = "main",
            backend: Backend = .gpu,
            visionBackend: Backend? = nil,
            audioBackend: Backend? = nil,
            maxTokens: Int = 2048,
            hub: HubClient? = nil,
            downloadProgress: Progress? = nil
        ) {
            self.imageSession = .shared
            self.engine = LiteRTModelLoader {
                guard let repo = Repo.ID(rawValue: huggingFaceRepo) else {
                    throw URLError(.badURL)
                }
                let destination = try await (hub ?? .default).downloadFile(
                    at: fileName,
                    from: repo,
                    revision: revision,
                    progress: downloadProgress
                )
                return try await makeEngine(
                    modelPath: destination.path,
                    backend: backend,
                    visionBackend: visionBackend,
                    audioBackend: audioBackend,
                    maxTokens: maxTokens
                )
            }
        }

        init(
            load: @escaping @Sendable () async throws -> any LiteRTRuntime,
            imageSession: URLSession = .shared
        ) {
            self.engine = LiteRTModelLoader(load)
            self.imageSession = imageSession
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
            var plan = try await makePlan(
                from: session.transcript,
                fallbackPrompt: prompt.description,
                schemaJSON: includeSchemaInPrompt ? schemaJSON : nil,
                tools: tools,
                imageSession: imageSession
            )
            let sampler = makeSampler(for: options, structured: schemaJSON != nil || !tools.isEmpty)

            var entries: [Transcript.Entry] = []
            var text = ""
            var toolRounds = 0

            while true {
                let conversation = try await engine.makeConversation(
                    config: ConversationConfig(
                        systemMessage: plan.systemMessage,
                        initialMessages: plan.history,
                        samplerConfig: sampler
                    )
                )

                text = ""
                try await generateLiteRTResponse(
                    conversation: conversation,
                    prompt: plan.prompt,
                    maximumResponseTokens: options.maximumResponseTokens
                ) { chunk in
                    text += chunk
                }

                guard !tools.isEmpty,
                    let parsed = parseToolCall(from: text, tools: tools)
                else { break }
                guard toolRounds < maxToolRounds else {
                    throw LanguageModelSession.GenerationError.decodingFailure(
                        .init(debugDescription: "Exceeded maximum LiteRT tool iterations (\(maxToolRounds)).")
                    )
                }
                toolRounds += 1

                let resolution = try await resolveToolCall(
                    name: parsed.name,
                    argumentsJSON: parsed.arguments,
                    session: session
                )
                switch resolution {
                case .stop(let call):
                    guard type == String.self else {
                        throw LanguageModelSession.GenerationError.decodingFailure(
                            .init(
                                debugDescription:
                                    "Tool execution stopped before LiteRT generated a structured response."
                            )
                        )
                    }
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

            guard let json = extractJSONValue(from: text) else {
                throw LanguageModelSession.GenerationError.decodingFailure(
                    .init(debugDescription: "LiteRT did not generate a complete JSON value.")
                )
            }
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

                            let plan = try await makePlan(
                                from: session.transcript,
                                fallbackPrompt: prompt.description,
                                schemaJSON: includeSchemaInPrompt ? schemaJSON : nil,
                                tools: [],
                                imageSession: imageSession
                            )
                            let conversation = try await engine.makeConversation(
                                config: ConversationConfig(
                                    systemMessage: plan.systemMessage,
                                    initialMessages: plan.history,
                                    samplerConfig: makeSampler(for: options, structured: schemaJSON != nil)
                                )
                            )

                            var text = ""
                            var lastJSON: String?
                            try await generateLiteRTResponse(
                                conversation: conversation,
                                prompt: plan.prompt,
                                maximumResponseTokens: options.maximumResponseTokens
                            ) { delta in
                                guard !delta.isEmpty else { return }
                                text += delta

                                if type == String.self {
                                    continuation.yield(
                                        .init(
                                            content: (text as! Content).asPartiallyGenerated(),
                                            rawContent: GeneratedContent(text)
                                        )
                                    )
                                } else if let json = extractJSONValue(from: text, isFinal: false),
                                    json != lastJSON,
                                    let raw = try? GeneratedContent(json: json),
                                    let parsed = try? type.init(raw)
                                {
                                    lastJSON = json
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

                            if type != String.self {
                                guard let json = extractJSONValue(from: text) else {
                                    throw LanguageModelSession.GenerationError.decodingFailure(
                                        .init(debugDescription: "LiteRT did not generate a complete JSON value.")
                                    )
                                }
                                if json != lastJSON {
                                    let raw = try GeneratedContent(json: json)
                                    let parsed = try type.init(raw)
                                    continuation.yield(
                                        .init(content: parsed.asPartiallyGenerated(), rawContent: raw)
                                    )
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

    private func makeEngine(
        modelPath: String,
        backend: Backend,
        visionBackend: Backend?,
        audioBackend: Backend?,
        maxTokens: Int
    ) async throws -> Engine {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        let config = try EngineConfig(
            modelPath: modelPath,
            backend: backend,
            visionBackend: visionBackend,
            audioBackend: audioBackend,
            maxNumTokens: maxTokens,
            cacheDir: caches?.path
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

        /// Extends the plan after a tool round-trip:
        /// the trigger prompt and the model's tool-call text become history,
        /// and the tool result becomes the new trigger.
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

    /// Splits the session transcript into a system message,
    /// prior turns, and the message to generate from.
    /// The generation trigger is the last `.prompt`
    /// or (in a tool round-trip) the last `.toolOutput` entry.
    private func makePlan(
        from transcript: Transcript,
        fallbackPrompt: String,
        schemaJSON: String?,
        tools: [any Tool],
        imageSession: URLSession
    ) async throws -> GenerationPlan {
        let entries = Array(transcript)
        let triggerIndex = entries.lastIndex { entry in
            switch entry {
            case .prompt, .toolOutput: return true
            default: return false
            }
        }

        var systemText: [String] = []
        let describedTools = tools.filter(\.includesSchemaInInstructions)
        if !describedTools.isEmpty {
            systemText.append(toolInstructions(describedTools))
        }
        var history: [Message] = []
        var trigger: Message?

        for (index, entry) in entries.enumerated() {
            let isTrigger = (index == triggerIndex)
            switch entry {
            case .instructions(let instructions):
                systemText.append(textContent(of: instructions.segments))
            case .prompt(let prompt):
                var contents = try await messageContents(of: prompt.segments, session: imageSession)
                if isTrigger, let schemaJSON, !schemaJSON.isEmpty {
                    contents.append(
                        .text(
                            "\n\nRespond with ONLY a JSON value that conforms to this JSON schema. "
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

    /// Maps transcript segments to LiteRT content:
    /// text, structured content as JSON text, and images.
    private func messageContents(
        of segments: [Transcript.Segment],
        session: URLSession
    ) async throws -> [Content] {
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
                    } else {
                        let (data, response) = try await session.data(from: url)
                        if let response = response as? HTTPURLResponse,
                            !(200 ... 299).contains(response.statusCode)
                        {
                            throw URLError(.badServerResponse)
                        }
                        contents.append(.imageData(data))
                    }
                }
            }
        }
        return contents.isEmpty ? [.text("")] : contents
    }

    /// Concatenates text and structured JSON from a segment list.
    /// Image segments are ignored.
    private func textContent(of segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            case .image: return nil
            }
        }.joined(separator: " ")
    }

    // MARK: - Structured Generation

    private func encodeSchema(_ schema: GenerationSchema) throws -> String {
        let resolvedSchema = schema.withResolvedRoot() ?? schema
        let data = try JSONEncoder().encode(resolvedSchema)
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// Extracts the first complete JSON value from model text,
    /// stripping surrounding prose and code fences.
    /// Scalars at the end of a stream must wait for the next delimiter or completion,
    /// because another chunk can extend a number or literal.
    private func extractJSONValue(from text: String, isFinal: Bool = true) -> String? {
        var start = text.startIndex
        while start < text.endIndex {
            let first = text[start]
            let isContainer = first == "{" || first == "["
            let isString = first == "\""
            let isBoundary =
                start == text.startIndex
                || !(text[text.index(before: start)].isLetter
                    || text[text.index(before: start)].isNumber
                    || text[text.index(before: start)] == ".")
            guard
                isContainer || isString
                    || (isBoundary && "-0123456789tfn".contains(first))
            else {
                start = text.index(after: start)
                continue
            }

            var end = start
            if isContainer || isString {
                var depth = 0
                var inString = false
                var escaped = false
                repeat {
                    let character = text[end]
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
                    } else if character == "{" || character == "[" {
                        depth += 1
                    } else if character == "}" || character == "]" {
                        depth -= 1
                    }
                    end = text.index(after: end)
                } while end < text.endIndex && (inString || depth > 0)
                guard !inString, depth == 0 else { return nil }
            } else {
                while end < text.endIndex,
                    text[end].isLetter || text[end].isNumber || ".+-".contains(text[end])
                {
                    end = text.index(after: end)
                }
                if end == text.endIndex && !isFinal { return nil }
            }

            let candidate = String(text[start ..< end])
            if (try? JSONSerialization.jsonObject(with: Data(candidate.utf8), options: .fragmentsAllowed)) != nil {
                return candidate
            }
            start = end
        }
        return nil
    }

    // MARK: - Sampling

    private func makeSampler(for options: GenerationOptions, structured: Bool) -> SamplerConfig? {
        var topK = 40
        var topP = 0.95
        var seed: UInt64 = 0
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
            case .topK(let k, let randomSeed):
                topK = min(k, Int(Int32.max))
                topP = 1.0
                seed = randomSeed ?? 0
            case .nucleus(let probabilityThreshold, let randomSeed):
                // LiteRT clamps top-k to the vocabulary size.
                topK = Int(Int32.max)
                topP = probabilityThreshold
                seed = randomSeed ?? 0
            }
        }
        // The Swift wrapper converts seeds to the runtime's signed 32-bit representation.
        // Preserve the low 32 bits without trapping on larger UInt64 values.
        return try? SamplerConfig(
            topK: topK,
            topP: Float(topP),
            temperature: Float(temperature),
            seed: Int(Int32(truncatingIfNeeded: seed))
        )
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

    /// Parses a tool call from model output,
    /// if present and naming a known tool.
    private func parseToolCall(
        from text: String,
        tools: [any Tool]
    ) -> (name: String, arguments: String)? {
        guard let start = text.firstIndex(of: "{"),
            let json = extractJSONValue(from: String(text[start...])),
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
