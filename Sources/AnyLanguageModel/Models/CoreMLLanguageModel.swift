#if CoreML
    import Foundation
    import CoreML
    import Tokenizers
    import JSONSchema
    @preconcurrency import Generation
    @preconcurrency import Models

    /// A language model that runs locally using Core ML.
    ///
    /// Use this model to run language models on-device with Core ML.
    /// The model must be compiled to `.mlmodelc` format before use.
    ///
    /// ```swift
    /// let modelURL = Bundle.main.url(
    ///     forResource: "MyModel",
    ///     withExtension: "mlmodelc"
    /// )!
    /// let model = try await CoreMLLanguageModel(url: modelURL)
    /// ```
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    public struct CoreMLLanguageModel: AnyLanguageModel.LanguageModel {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        private let model: Models.LanguageModel
        private let tokenizer: any Tokenizer
        private let chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])?
        private let toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])?

        /// Creates a Core ML language model.
        ///
        /// - Parameters:
        ///   - url: The URL to a compiled Core ML model (`.mlmodelc`).
        ///   - computeUnits: The compute units to use for inference.
        ///   - chatTemplateHandler: An optional handler to format chat messages.
        ///   - toolsHandler: An optional handler to convert tools to the model's expected format.
        ///
        /// - Throws: A `CoreMLLanguageModelError` if the model can't be loaded, the file doesn't exist, or the model is invalid.
        public init(
            url: URL,
            computeUnits: MLComputeUnits = .all,
            chatTemplateHandler: (@Sendable (Instructions?, Prompt) -> [Message])? = nil,
            toolsHandler: (@Sendable ([any Tool]) -> [ToolSpec])? = nil
        ) async throws {
            // Ensure the model is already compiled
            guard url.pathExtension == "mlmodelc" else {
                throw CoreMLLanguageModelError.compiledModelRequired
            }

            // Check if the file exists first
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw CoreMLLanguageModelError.modelNotFound(url)
            }

            do {
                // Load the model with the specified compute units
                self.model = try Models.LanguageModel.loadCompiled(url: url, computeUnits: computeUnits)
            } catch {
                // Map CoreML errors to our specific error cases
                throw CoreMLLanguageModelError.modelInvalid(url, underlyingError: error)
            }

            // Load the tokenizer
            self.tokenizer = try await model.tokenizer

            self.chatTemplateHandler = chatTemplateHandler
            self.toolsHandler = toolsHandler
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            try validateNoImageSegments(in: session)

            if type != String.self {
                let jsonString = try await generateStructuredJSON(
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

            // Convert AnyLanguageModel GenerationOptions to swift-transformers GenerationConfig
            let generationConfig = toGenerationConfig(options)

            let tokens: [Int]
            if let chatTemplateHandler = chatTemplateHandler {
                // Use chat template handler with optional tools
                let messages = chatTemplateHandler(session.instructions, prompt)
                let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                tokens = try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            } else {
                // Fall back to direct tokenizer encoding
                tokens = tokenizer.encode(text: prompt.description)
            }

            // Reset model state for new generation
            await model.resetState()

            let outputTokens = await model.generate(
                config: generationConfig,
                tokens: tokens,
                model: model.callAsFunction
            )

            // Strip the prompt at the token level to avoid issues with
            // normalization or whitespace differences in decoded strings
            let assistantTokenSlice: ArraySlice<Int>
            if outputTokens.count >= tokens.count {
                assistantTokenSlice = outputTokens.dropFirst(tokens.count)
            } else {
                // Fallback: if the model did not echo the full prompt,
                // treat the entire output as assistant tokens
                assistantTokenSlice = outputTokens[outputTokens.indices]
            }
            let assistantText = tokenizer.decode(tokens: Array(assistantTokenSlice))

            return LanguageModelSession.Response(
                content: assistantText as! Content,
                rawContent: GeneratedContent(assistantText),
                transcriptEntries: ArraySlice([])
            )
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            // For now, only String is supported
            guard type == String.self else {
                return LanguageModelSession.ResponseStream(
                    stream: AsyncThrowingStream { continuation in
                        continuation.finish(
                            throwing: CoreMLLanguageModelError.structuredStreamingUnsupported
                        )
                    }
                )
            }

            // Validate that no image segments are present
            do {
                try validateNoImageSegments(in: session)
            } catch {
                return LanguageModelSession.ResponseStream(
                    stream: AsyncThrowingStream { continuation in
                        continuation.finish(throwing: error)
                    }
                )
            }

            // Convert AnyLanguageModel GenerationOptions to swift-transformers GenerationConfig
            let generationConfig = toGenerationConfig(options)

            // Transform the generation into ResponseStream snapshots
            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                @Sendable continuation in
                let task = Task {
                    do {
                        let tokens: [Int]
                        if let chatTemplateHandler = chatTemplateHandler {
                            // Use chat template handler with optional tools
                            let messages = chatTemplateHandler(session.instructions, prompt)
                            let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                            tokens = try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
                        } else {
                            // Fall back to direct tokenizer encoding
                            tokens = tokenizer.encode(text: prompt.description)
                        }

                        await model.resetState()

                        let promptTokenCount = tokens.count
                        var accumulatedText = ""

                        _ = await model.generate(
                            config: generationConfig,
                            tokens: tokens,
                            model: model.callAsFunction
                        ) { tokenIds in
                            let assistantTokenSlice: ArraySlice<Int>
                            if tokenIds.count >= promptTokenCount {
                                assistantTokenSlice = tokenIds.dropFirst(promptTokenCount)
                            } else {
                                assistantTokenSlice = tokenIds[tokenIds.indices]
                            }
                            let assistantText = tokenizer.decode(tokens: Array(assistantTokenSlice))

                            // Compute delta vs accumulated text and yield
                            if assistantText.count >= accumulatedText.count,
                                assistantText.hasPrefix(accumulatedText)
                            {
                                let startIdx = assistantText.index(
                                    assistantText.startIndex,
                                    offsetBy: accumulatedText.count
                                )
                                let delta = String(assistantText[startIdx...])
                                accumulatedText += delta
                            } else {
                                accumulatedText = assistantText
                            }

                            continuation.yield(
                                .init(
                                    content: (accumulatedText as! Content).asPartiallyGenerated(),
                                    rawContent: GeneratedContent(accumulatedText)
                                )
                            )
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

        // MARK: - Image Validation

        private func validateNoImageSegments(in session: LanguageModelSession) throws {
            // Note: Instructions is a plain text type without segments, so no image check needed there.
            // Check for image segments in the most recent prompt
            for entry in session.transcript.reversed() {
                if case .prompt(let p) = entry {
                    for segment in p.segments {
                        if case .image = segment {
                            throw CoreMLLanguageModelError.unsupportedFeature
                        }
                    }
                    break
                }
            }
        }
    }

    /// Errors that can occur when working with Core ML language models.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    public enum CoreMLLanguageModelError: LocalizedError {
        /// The provided model isn't a compiled Core ML model.
        case compiledModelRequired

        /// The model file was not found at the specified URL.
        case modelNotFound(URL)

        /// The model file was found but is corrupted, incompatible, or otherwise invalid.
        case modelInvalid(URL, underlyingError: Error)
        /// Image segments are not supported in CoreMLLanguageModel
        case unsupportedFeature
        /// Structured response streaming is not supported in CoreMLLanguageModel
        case structuredStreamingUnsupported

        public var errorDescription: String? {
            switch self {
            case .compiledModelRequired:
                return
                    "A compiled Core ML model (.mlmodelc) is required. Please compile your model first using MLModel.compileModel(at:)."
            case .modelNotFound(let url):
                return "Core ML model not found at: \(url.path). Please verify the file exists and the path is correct."
            case .modelInvalid(let url, let underlyingError):
                return
                    "Core ML model at \(url.path) is invalid or corrupted: \(underlyingError.localizedDescription). Please verify the model file is valid and compatible with the current Core ML version."
            case .unsupportedFeature:
                return "This CoreMLLanguageModel does not support image segments"
            case .structuredStreamingUnsupported:
                return "This CoreMLLanguageModel does not support structured response streaming"
            }
        }
    }

    // MARK: -

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    extension CoreMLLanguageModel {
        private func toGenerationConfig(_ options: GenerationOptions) -> GenerationConfig {
            var config = GenerationConfig(maxNewTokens: options.maximumResponseTokens ?? 2048)

            // Map temperature
            if let temperature = options.temperature {
                config.temperature = Float(temperature)
            }

            // Map sampling mode
            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    config.doSample = false
                case .topK(let k, _):
                    config.doSample = true
                    config.topK = k
                case .nucleus(let p, _):
                    config.doSample = true
                    config.topP = Float(p)
                }
            }

            return config
        }

        private func toStructuredGenerationConfig(_ options: GenerationOptions) -> GenerationConfig {
            var config = GenerationConfig(maxNewTokens: options.maximumResponseTokens ?? 512)

            config.doSample = true
            if let temperature = options.temperature {
                config.temperature = Float(temperature)
            } else {
                config.temperature = 0.2
            }
            config.topP = 0.95
            config.repetitionPenalty = 1.1

            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    config.doSample = false
                case .topK(let k, _):
                    config.doSample = true
                    config.topK = k
                case .nucleus(let p, _):
                    config.doSample = true
                    config.topP = Float(p)
                }
            }

            return config
        }

        private func generateStructuredJSON(
            session: LanguageModelSession,
            prompt: Prompt,
            schema: GenerationSchema,
            options: GenerationOptions,
            includeSchemaInPrompt: Bool
        ) async throws -> String {
            let maxTokens = options.maximumResponseTokens ?? 512
            var generationConfig = toStructuredGenerationConfig(options)

            let promptTokens = try structuredPromptTokens(
                in: session,
                prompt: prompt,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt
            )

            generationConfig.maxLength = generationConfig.maxNewTokens + promptTokens.count
            generationConfig.eosTokenId = tokenizer.eosTokenId
            generationConfig.bosTokenId = tokenizer.bosTokenId

            await model.resetState()

            let tokenTensor = MLTensor(promptTokens.map(Int32.init)).expandingShape(at: 0)
            let initialLogits = await model.predictNextTokenScores(tokenTensor, config: generationConfig)
            let endTokens: Set<Int> = []

            let backend = try CoreMLTokenBackend(
                model: model,
                tokenizer: tokenizer,
                config: generationConfig,
                tokens: promptTokens,
                initialLogits: initialLogits,
                maximumTokens: maxTokens,
                endTokens: endTokens
            )
            var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
            let json = try await generator.generate()
            return json
        }

        private func structuredPromptTokens(
            in session: LanguageModelSession,
            prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool
        ) throws -> [Int] {
            if let chatTemplateHandler = chatTemplateHandler {
                var messages = chatTemplateHandler(session.instructions, prompt)
                if includeSchemaInPrompt {
                    let schemaPrompt = schemaPrompt(for: schema)
                    if !schemaPrompt.isEmpty {
                        messages.insert(["role": "system", "content": schemaPrompt], at: 0)
                    }
                }
                let toolSpecs: [ToolSpec]? = toolsHandler?(session.tools)
                return try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            }

            var text = prompt.description
            if includeSchemaInPrompt {
                let schemaPrompt = schemaPrompt(for: schema)
                if !schemaPrompt.isEmpty {
                    text = "\(schemaPrompt)\n\n\(text)"
                }
            }
            return tokenizer.encode(text: text)
        }

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

        private struct CoreMLTokenBackend: TokenBackend {
            let model: Models.LanguageModel
            let tokenizer: any Tokenizer
            let config: GenerationConfig
            let logitsProcessorList: LogitsProcessorList
            let endTokens: Set<Int>
            let eosToken: Int
            let vocabSize: Int

            var tokens: [Int]
            var currentLogits: MLTensor
            var remainingTokens: Int
            let totalTokenBudget: Int

            init(
                model: Models.LanguageModel,
                tokenizer: any Tokenizer,
                config: GenerationConfig,
                tokens: [Int],
                initialLogits: MLTensor,
                maximumTokens: Int,
                endTokens: Set<Int>
            ) throws {
                self.model = model
                self.tokenizer = tokenizer
                self.config = config
                self.tokens = tokens
                self.currentLogits = initialLogits
                self.remainingTokens = maximumTokens
                self.totalTokenBudget = maximumTokens
                self.endTokens = endTokens
                self.eosToken = config.eosTokenId ?? tokenizer.eosTokenId ?? 0
                self.vocabSize = initialLogits.shape.last ?? 0
                self.logitsProcessorList = CoreMLLanguageModel.makeLogitsProcessorList(config: config)
            }

            func tokenize(_ text: String) throws -> [Int] {
                tokenizer.encode(text: text, addSpecialTokens: false)
            }

            func tokenText(_ token: Int) -> String? {
                let decoded = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
                return decoded.isEmpty ? nil : decoded
            }

            func isSpecialToken(_ token: Int) -> Bool {
                let raw = tokenizer.decode(tokens: [token], skipSpecialTokens: false)
                guard !raw.isEmpty else { return false }
                let filtered = tokenizer.decode(tokens: [token], skipSpecialTokens: true)
                return filtered.isEmpty
            }

            mutating func decode(_ token: Int) async throws {
                tokens.append(token)
                remainingTokens -= 1
                let tokenTensor = MLTensor(tokens.map(Int32.init)).expandingShape(at: 0)
                currentLogits = await model.predictNextTokenScores(tokenTensor, config: config)
            }

            mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
                guard !allowedTokens.isEmpty else {
                    throw ConstrainedGenerationError.tokenizationFailed
                }

                // Run logits processors on Float32 scores for stable behavior
                let inputIds = MLTensor(tokens.map(Int32.init)).expandingShape(at: 0)
                let floatScores =
                    currentLogits.scalarType == Float.self
                    ? currentLogits
                    : currentLogits.cast(to: Float.self)
                let vocabSize = floatScores.shape.last ?? self.vocabSize

                // Build a mask tensor that keeps only the allowed tokens.
                var maskValues = Array(repeating: -Float.infinity, count: vocabSize)
                var hasValidToken = false
                for token in allowedTokens {
                    if token >= 0 && token < vocabSize {
                        maskValues[token] = 0
                        hasValidToken = true
                    }
                }
                guard hasValidToken else {
                    throw ConstrainedGenerationError.tokenizationFailed
                }
                let maskTensor = MLTensor(maskValues).reshaped(to: floatScores.shape)
                let maskedScores = floatScores + maskTensor
                let processedScores = await logitsProcessorList(inputIds, maskedScores)

                let tokenTensor: MLTensor
                if config.doSample {
                    // Multinomial sample from candidate probabilities
                    let probs = processedScores.softmax(alongAxis: -1)
                    let prefixShape = Array(processedScores.shape.dropLast())
                    let randomShape = prefixShape + [1]
                    let rndTensor = MLTensor(randomUniform: randomShape, in: 0 ..< 1, scalarType: Float.self)
                    let cumulativeProbs = probs.cumulativeSum(alongAxis: -1)
                    let rnd =
                        cumulativeProbs.scalarType == Float.self
                        ? rndTensor : rndTensor.cast(to: cumulativeProbs.scalarType)

                    let mask = cumulativeProbs .< rnd
                    let penalized = mask * 1000.0
                    let indexed = penalized + cumulativeProbs
                    let sampledIndex = indexed.argmin(alongAxis: -1)
                    tokenTensor =
                        sampledIndex.scalarType == Int32.self ? sampledIndex : sampledIndex.cast(to: Int32.self)
                } else {
                    // Greedy select the best-scoring candidate
                    let selectedIndex = processedScores.argmax(alongAxis: -1)
                    tokenTensor =
                        selectedIndex.scalarType == Int32.self ? selectedIndex : selectedIndex.cast(to: Int32.self)
                }

                // Materialize the chosen token id
                let tokenArray = await tokenTensor.shapedArray(of: Int32.self)
                guard let token = tokenArray.scalars.last else {
                    throw ConstrainedGenerationError.tokenizationFailed
                }

                return Int(token)
            }
        }

        fileprivate static func makeLogitsProcessorList(config: GenerationConfig) -> LogitsProcessorList {
            var processors: [any LogitsProcessor] = []

            if config.repetitionPenalty != 1.0 {
                if let processor = try? RepetitionPenaltyLogitsProcessor(penalty: Float(config.repetitionPenalty)) {
                    processors.append(processor)
                }
            }

            if config.temperature > 0 && config.temperature != 1.0 {
                if let processor = try? TemperatureLogitsWarper(temperature: config.temperature) {
                    processors.append(processor)
                }
            }

            if config.topK > 0 && config.topK < Int.max {
                if let processor = try? TopKLogitsWarper(topK: config.topK) {
                    processors.append(processor)
                }
            }

            if config.topP < 1.0 {
                if let processor = try? TopPLogitsWarper(topP: Float(config.topP)) {
                    processors.append(processor)
                }
            }

            if let minP = config.minP {
                if let processor = try? MinPLogitsWarper(minP: Float(minP)) {
                    processors.append(processor)
                }
            }

            return LogitsProcessorList(processors: processors)
        }

    }
#endif  // CoreML
