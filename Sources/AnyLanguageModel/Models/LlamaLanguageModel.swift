import Foundation
#if Llama
    import JSONSchema
    import LlamaSwift
    import XGrammar

    /// Global storage for the current log level threshold.
    /// This is needed because the C callback can't capture Swift context.
    /// Access is synchronized by llama.cpp's internal logging mechanism.
    nonisolated(unsafe) private var currentLogLevel: LlamaLanguageModel.LogLevel = .warn

    /// Custom log callback that filters messages based on the current log level.
    private func llamaLogCallback(
        level: ggml_log_level,
        text: UnsafePointer<CChar>?,
        userData: UnsafeMutableRawPointer?
    ) {
        guard level.rawValue >= currentLogLevel.ggmlLevel.rawValue else { return }
        if let text = text {
            fputs(String(cString: text), stderr)
        }
    }

    /// A language model that runs llama.cpp models locally.
    ///
    /// Use this model to generate text using GGUF models running directly with llama.cpp.
    ///
    /// ```swift
    /// let model = LlamaLanguageModel(
    ///     modelPath: "/path/to/model.gguf",
    ///     contextSize: 2048
    /// )
    /// ```
    public final class LlamaLanguageModel: LanguageModel, @unchecked Sendable {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        /// The verbosity level for llama.cpp logging.
        public enum LogLevel: Int, Hashable, Comparable, Sendable, CaseIterable {
            /// No logging output.
            case none = 0
            /// Debug messages and above (most verbose).
            case debug = 1
            /// Info messages and above.
            case info = 2
            /// Warning messages and above (default).
            case warn = 3
            /// Only error messages.
            case error = 4

            /// Maps to the corresponding ggml log level.
            var ggmlLevel: ggml_log_level {
                switch self {
                case .none: return GGML_LOG_LEVEL_NONE
                case .debug: return GGML_LOG_LEVEL_DEBUG
                case .info: return GGML_LOG_LEVEL_INFO
                case .warn: return GGML_LOG_LEVEL_WARN
                case .error: return GGML_LOG_LEVEL_ERROR
                }
            }

            public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
                lhs.rawValue < rhs.rawValue
            }
        }

        /// Custom generation options specific to llama.cpp.
        ///
        /// Use this type to pass llama.cpp-specific sampling parameters that are
        /// not part of the standard ``GenerationOptions``.
        ///
        /// ```swift
        /// var options = GenerationOptions(temperature: 0.8)
        /// options[custom: LlamaLanguageModel.self] = .init(
        ///     repeatPenalty: 1.2,
        ///     repeatLastN: 128,
        ///     frequencyPenalty: 0.1,
        ///     presencePenalty: 0.1,
        ///     mirostat: .v2(tau: 5.0, eta: 0.1)
        /// )
        /// ```
        public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions, Codable {
            /// Context size to allocate for the model.
            public var contextSize: UInt32?

            /// Batch size to use when evaluating tokens.
            public var batchSize: UInt32?

            /// Number of threads to use for computation.
            public var threads: Int32?

            /// Random seed for deterministic sampling.
            public var seed: UInt32?

            /// Sampling temperature.
            public var temperature: Float?

            /// Top-K sampling parameter.
            public var topK: Int32?

            /// Top-P (nucleus) sampling parameter.
            public var topP: Float?

            /// The penalty applied to repeated tokens.
            ///
            /// Values greater than 1.0 discourage repetition, while values less than 1.0
            /// encourage it. A value of 1.0 applies no penalty.
            public var repeatPenalty: Float?

            /// The number of recent tokens to consider for the repeat penalty.
            ///
            /// Only the last `repeatLastN` tokens will be checked for repetition.
            /// Set to 0 to disable repeat penalty, or -1 to consider all tokens.
            public var repeatLastN: Int32?

            /// The frequency penalty applied during sampling.
            ///
            /// Positive values penalize tokens based on their frequency in the text so far,
            /// decreasing the likelihood of repeating the same content.
            public var frequencyPenalty: Float?

            /// The presence penalty applied during sampling.
            ///
            /// Positive values penalize tokens that have appeared at all in the text so far,
            /// encouraging the model to generate novel content.
            public var presencePenalty: Float?

            /// Mirostat sampling configuration for adaptive perplexity control.
            public enum MirostatMode: Hashable, Codable, Sendable {
                /// Mirostat v1 with target entropy (tau) and learning rate (eta).
                case v1(tau: Float, eta: Float)

                /// Mirostat v2 with target entropy (tau) and learning rate (eta).
                case v2(tau: Float, eta: Float)
            }

            /// Mirostat sampling mode for adaptive perplexity control.
            public var mirostat: MirostatMode?

            /// Creates custom generation options for llama.cpp.
            public init(
                contextSize: UInt32? = nil,
                batchSize: UInt32? = nil,
                threads: Int32? = nil,
                seed: UInt32? = nil,
                temperature: Float? = nil,
                topK: Int32? = nil,
                topP: Float? = nil,
                repeatPenalty: Float? = nil,
                repeatLastN: Int32? = nil,
                frequencyPenalty: Float? = nil,
                presencePenalty: Float? = nil,
                mirostat: MirostatMode? = nil
            ) {
                self.contextSize = contextSize
                self.batchSize = batchSize
                self.threads = threads
                self.seed = seed
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
                self.repeatPenalty = repeatPenalty
                self.repeatLastN = repeatLastN
                self.frequencyPenalty = frequencyPenalty
                self.presencePenalty = presencePenalty
                self.mirostat = mirostat
            }

            /// Default llama.cpp options used when none are provided at runtime.
            ///
            /// The `seed` is `nil` by default, meaning a random seed will be generated
            /// for each generation request.
            public static var `default`: Self {
                .init(
                    contextSize: 2048,
                    batchSize: 512,
                    threads: Int32(ProcessInfo.processInfo.processorCount),
                    seed: nil,
                    temperature: 0.8,
                    topK: 40,
                    topP: 0.95,
                    repeatPenalty: 1.1,
                    repeatLastN: 64,
                    frequencyPenalty: 0.0,
                    presencePenalty: 0.0,
                    mirostat: nil
                )
            }

        }

        /// The path to the GGUF model file.
        public let modelPath: String

        /// The context size for the model.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var contextSize: UInt32 { legacyDefaults.contextSize }

        /// The batch size for processing.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(batchSize: 1024)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var batchSize: UInt32 { legacyDefaults.batchSize }

        /// The number of threads to use.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(threads: 8)
        ///   ```
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var threads: Int32 { legacyDefaults.threads }

        /// The random seed for generation.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(seed: 42)
        ///   ```
        ///   custom options instead.
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var seed: UInt32 { legacyDefaults.seed }

        /// The temperature for sampling.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(temperature: 0.6)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var temperature: Float { legacyDefaults.temperature }

        /// The top-K sampling parameter.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(topK: 25)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var topK: Int32 { legacyDefaults.topK }

        /// The top-P (nucleus) sampling parameter.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(topP: 0.9)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var topP: Float { legacyDefaults.topP }

        /// The repeat penalty for generation.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(repeatPenalty: 1.2)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var repeatPenalty: Float { legacyDefaults.repeatPenalty }

        /// The number of tokens to consider for repeat penalty.
        ///
        /// - Important: This property is deprecated.
        ///   Use ``GenerationOptions`` with custom options instead:
        ///   ```swift
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(repeatLastN: 128)
        ///   ```
        @available(*, deprecated, message: "Use GenerationOptions custom options instead")
        public var repeatLastN: Int32 { legacyDefaults.repeatLastN }

        /// Normalized legacy defaults used for deprecated properties.
        private let legacyDefaults: ResolvedGenerationOptions

        /// The minimum log level for llama.cpp output.
        ///
        /// This is a global setting that affects all `LlamaLanguageModel` instances
        /// since llama.cpp uses a single global log callback.
        public nonisolated(unsafe) static var logLevel: LogLevel = .warn {
            didSet {
                currentLogLevel = logLevel
                llama_log_set(llamaLogCallback, nil)
            }
        }

        /// Resolved, non-optional defaults for llama.cpp runtime parameters.
        internal struct ResolvedGenerationOptions: Sendable {
            var contextSize: UInt32
            var batchSize: UInt32
            var threads: Int32
            var seed: UInt32
            var temperature: Float
            var topK: Int32
            var topP: Float
            var repeatPenalty: Float
            var repeatLastN: Int32
            var frequencyPenalty: Float
            var presencePenalty: Float
            var mirostat: CustomGenerationOptions.MirostatMode?
            var sampling: GenerationOptions.SamplingMode?
            var maximumResponseTokens: Int?

            init(
                contextSize: UInt32 = 2048,
                batchSize: UInt32 = 512,
                threads: Int32 = Int32(ProcessInfo.processInfo.processorCount),
                seed: UInt32 = UInt32.random(in: 0 ... UInt32.max),
                temperature: Float = 0.8,
                topK: Int32 = 40,
                topP: Float = 0.95,
                repeatPenalty: Float = 1.1,
                repeatLastN: Int32 = 64,
                frequencyPenalty: Float = 0.0,
                presencePenalty: Float = 0.0,
                mirostat: CustomGenerationOptions.MirostatMode? = nil,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                self.contextSize = contextSize
                self.batchSize = batchSize
                self.threads = threads
                self.seed = seed
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
                self.repeatPenalty = repeatPenalty
                self.repeatLastN = repeatLastN
                self.frequencyPenalty = frequencyPenalty
                self.presencePenalty = presencePenalty
                self.mirostat = mirostat
                self.sampling = sampling
                self.maximumResponseTokens = maximumResponseTokens
            }

            init(
                from options: CustomGenerationOptions?,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                self.init(
                    base: ResolvedGenerationOptions(),
                    overrides: options,
                    sampling: sampling,
                    maximumResponseTokens: maximumResponseTokens
                )
            }

            init(
                base: ResolvedGenerationOptions = .init(),
                overrides options: CustomGenerationOptions?,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                guard let options else {
                    self = ResolvedGenerationOptions(
                        contextSize: base.contextSize,
                        batchSize: base.batchSize,
                        threads: base.threads,
                        seed: base.seed,
                        temperature: base.temperature,
                        topK: base.topK,
                        topP: base.topP,
                        repeatPenalty: base.repeatPenalty,
                        repeatLastN: base.repeatLastN,
                        frequencyPenalty: base.frequencyPenalty,
                        presencePenalty: base.presencePenalty,
                        mirostat: base.mirostat,
                        sampling: sampling ?? base.sampling,
                        maximumResponseTokens: maximumResponseTokens ?? base.maximumResponseTokens
                    )
                    return
                }

                self.contextSize = options.contextSize ?? base.contextSize
                self.batchSize = options.batchSize ?? base.batchSize
                self.threads = options.threads ?? base.threads
                self.seed = options.seed ?? base.seed
                self.temperature = options.temperature ?? base.temperature
                self.topK = options.topK ?? base.topK
                self.topP = options.topP ?? base.topP
                self.repeatPenalty = options.repeatPenalty ?? base.repeatPenalty
                self.repeatLastN = options.repeatLastN ?? base.repeatLastN
                self.frequencyPenalty = options.frequencyPenalty ?? base.frequencyPenalty
                self.presencePenalty = options.presencePenalty ?? base.presencePenalty
                self.mirostat = options.mirostat ?? base.mirostat
                self.sampling = sampling ?? base.sampling
                self.maximumResponseTokens = maximumResponseTokens ?? base.maximumResponseTokens
            }
        }

        /// The loaded model instance
        private var model: OpaquePointer?

        /// The model's vocabulary
        private var vocab: OpaquePointer?

        /// Whether the model is currently loaded
        private var isModelLoaded: Bool = false

        /// Creates a Llama language model.
        ///
        /// - Parameters:
        ///   - modelPath: The path to the GGUF model file.
        public init(modelPath: String) {
            self.modelPath = modelPath
            self.legacyDefaults = ResolvedGenerationOptions()
        }

        /// Creates a Llama language model using legacy parameter defaults.
        ///
        /// - Important: This initializer is deprecated.
        ///   Use `init(modelPath:)` and configure per-request values via
        ///   ``GenerationOptions`` custom options instead.
        ///
        ///   ```swift
        ///   let model = LlamaLanguageModel(modelPath: "/path/to/model.gguf")
        ///   var options = GenerationOptions()
        ///   options[custom: LlamaLanguageModel.self] = .init(contextSize: 4096)
        ///
        ///   let session = LanguageModelSession(model: model)
        ///   session.respond(to: "Hello, world!", options: options)
        ///   ```
        @available(
            *,
            deprecated,
            message: "Use init(modelPath:) and pass values via GenerationOptions custom options"
        )
        public convenience init(
            modelPath: String,
            contextSize: UInt32 = 2048,
            batchSize: UInt32 = 512,
            threads: Int32 = Int32(ProcessInfo.processInfo.processorCount),
            seed: UInt32 = UInt32.random(in: 0 ... UInt32.max),
            temperature: Float = 0.8,
            topK: Int32 = 40,
            topP: Float = 0.95,
            repeatPenalty: Float = 1.1,
            repeatLastN: Int32 = 64
        ) {
            // Deprecated: prefer setting these via GenerationOptions custom options.
            // We intentionally ignore legacy parameters to avoid storing model-level state.
            self.init(modelPath: modelPath)
        }

        deinit {
            if let model = model {
                llama_model_free(model)
            }
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            // Validate that no image segments are present
            try validateNoImageSegments(in: session)
            try await ensureModelLoaded()

            let runtimeOptions = resolvedOptions(from: options)
            let structuredOptions = resolvedStructuredOptions(from: options)
            let contextParams = createContextParams(from: runtimeOptions)

            // Try to create context with error handling
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }
            defer { llama_free(context) }

            // Check if this is an embedding model (no KV cache).
            // This early check catches models configured for embeddings that lack a KV cache.
            // A complementary architectural check in prepareInitialBatch catches encoder-only
            // models (like BERT) by their architecture type.
            if llama_get_memory(context) == nil {
                throw LlamaLanguageModelError.encoderOnlyModel
            }

            llama_set_causal_attn(context, true)
            llama_set_warmup(context, false)
            llama_set_n_threads(context, runtimeOptions.threads, runtimeOptions.threads)

            let fullPrompt: String
            if includeSchemaInPrompt, type != String.self {
                fullPrompt = try formatPrompt(
                    for: session,
                    extraSystemMessage: schemaPrompt(for: type.generationSchema)
                )
            } else {
                fullPrompt = try formatPrompt(for: session)
            }

            if type == String.self {
                let maxTokens = runtimeOptions.maximumResponseTokens ?? 100
                let text = try await generateText(
                    context: context,
                    model: model!,
                    prompt: fullPrompt,
                    maxTokens: maxTokens,
                    options: runtimeOptions
                )

                return LanguageModelSession.Response(
                    content: text as! Content,
                    rawContent: GeneratedContent(text),
                    transcriptEntries: ArraySlice([])
                )
            } else {
                let maxTokens = structuredOptions.maximumResponseTokens ?? 512
                let jsonString = try await generateStructuredJSON(
                    context: context,
                    prompt: fullPrompt,
                    schema: type.generationSchema,
                    maxTokens: maxTokens,
                    options: structuredOptions
                )
                let generatedContent = try GeneratedContent(json: jsonString)
                let content = try type.init(generatedContent)
                return LanguageModelSession.Response(
                    content: content,
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice([])
                )
            }
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
                fatalError("LlamaLanguageModel only supports generating String content")
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

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            try await ensureModelLoaded()

                            let runtimeOptions = resolvedOptions(from: options)
                            let maxTokens = runtimeOptions.maximumResponseTokens ?? 100
                            let contextParams = createContextParams(from: runtimeOptions)
                            guard let context = llama_init_from_model(model!, contextParams) else {
                                throw LlamaLanguageModelError.contextInitializationFailed
                            }
                            defer { llama_free(context) }

                            // Check if this is an embedding model (no KV cache).
                            // This early check catches models configured for embeddings that lack a KV cache.
                            // A complementary architectural check in prepareInitialBatch catches encoder-only
                            // models (like BERT) by their architecture type.
                            if llama_get_memory(context) == nil {
                                throw LlamaLanguageModelError.encoderOnlyModel
                            }

                            // Stabilize runtime behavior per-context
                            llama_set_causal_attn(context, true)
                            llama_set_warmup(context, false)
                            llama_set_n_threads(context, runtimeOptions.threads, runtimeOptions.threads)

                            var accumulatedText = ""
                            let fullPrompt = try self.formatPrompt(for: session)

                            do {
                                for try await tokenText in generateTextStream(
                                    context: context,
                                    model: model!,
                                    prompt: fullPrompt,
                                    maxTokens: maxTokens,
                                    options: runtimeOptions
                                ) {
                                    accumulatedText += tokenText

                                    let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                                        content: (accumulatedText as! Content).asPartiallyGenerated(),
                                        rawContent: GeneratedContent(accumulatedText)
                                    )
                                    continuation.yield(snapshot)
                                }
                            } catch {
                                continuation.finish(throwing: error)
                                return
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

        // MARK: - Private Helpers

        private func ensureModelLoaded() async throws {
            guard !isModelLoaded else { return }

            // Check if model file exists
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw LlamaLanguageModelError.invalidModelPath
            }

            // Initialize backend lazily - must be done before loading model
            llama_backend_init()

            // Free any existing model before loading a new one
            if let existingModel = model {
                llama_model_free(existingModel)
                self.model = nil
            }

            let modelParams = createModelParams()
            guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            self.model = loadedModel
            self.vocab = llama_model_get_vocab(loadedModel)
            self.isModelLoaded = true
        }

        private func createModelParams() -> llama_model_params {
            var params = llama_model_default_params()

            // Force CPU-only execution to avoid Metal GPU issues
            params.n_gpu_layers = 0

            // Try to reduce memory usage
            params.use_mmap = true
            params.use_mlock = false
            return params
        }

        private func resolvedOptions(from options: GenerationOptions) -> ResolvedGenerationOptions {
            var base = legacyDefaults
            if let temp = options.temperature {
                base.temperature = Float(temp)
            }

            return ResolvedGenerationOptions(
                base: base,
                overrides: options[custom: LlamaLanguageModel.self],
                sampling: options.sampling,
                maximumResponseTokens: options.maximumResponseTokens
            )
        }

        /// Builds structured-generation defaults while honoring explicit overrides.
        private func resolvedStructuredOptions(from options: GenerationOptions) -> ResolvedGenerationOptions {
            var base = legacyDefaults
            if let temp = options.temperature {
                base.temperature = Float(temp)
            } else {
                base.temperature = 0.2
            }
            base.topP = 0.95
            base.repeatPenalty = 1.1
            base.repeatLastN = 64

            return ResolvedGenerationOptions(
                base: base,
                overrides: options[custom: LlamaLanguageModel.self],
                sampling: options.sampling,
                maximumResponseTokens: options.maximumResponseTokens
            )
        }

        private func createContextParams(from options: ResolvedGenerationOptions) -> llama_context_params {
            var params = llama_context_default_params()
            params.n_ctx = options.contextSize
            params.n_batch = options.batchSize
            params.n_threads = options.threads
            params.n_threads_batch = options.threads
            return params
        }

        private func applySampling(
            sampler: UnsafeMutablePointer<llama_sampler>,
            effectiveTemperature: Float,
            options: ResolvedGenerationOptions
        ) {
            if let mirostat = options.mirostat {
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))

                switch mirostat {
                case .v1(let tau, let eta):
                    llama_sampler_chain_add(
                        sampler,
                        llama_sampler_init_mirostat(
                            Int32(options.contextSize),
                            options.seed,
                            tau,
                            eta,
                            100
                        )
                    )
                case .v2(let tau, let eta):
                    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(options.seed, tau, eta))
                }
                return
            }

            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
                case .topK(let k, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(k)))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
                    let samplingSeed = seed.map(UInt32.init) ?? options.seed
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingSeed))
                case .nucleus(let threshold, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(0))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(Float(threshold), 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
                    let samplingSeed = seed.map(UInt32.init) ?? options.seed
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingSeed))
                }
                return
            }

            if options.topK > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.topK))
            }
            if options.topP < 1.0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.topP, 1))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed))
        }

        private func generateText(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: ResolvedGenerationOptions
        ) async throws
            -> String
        {
            guard let vocab = llama_model_get_vocab(model) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            // Tokenize the prompt
            let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
            guard !promptTokens.isEmpty else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
            defer { llama_batch_free(batch) }

            let hasEncoder = try prepareInitialBatch(
                batch: &batch,
                promptTokens: promptTokens,
                model: model,
                vocab: vocab,
                context: context,
                batchSize: options.batchSize
            )

            // Initialize sampler chain with options
            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }
            let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

            let effectiveTemperature = Float(options.temperature)

            // Apply repeat/frequency/presence penalties from custom options
            let effectiveRepeatPenalty = options.repeatPenalty
            let effectiveRepeatLastN = options.repeatLastN
            let effectiveFrequencyPenalty = options.frequencyPenalty
            let effectivePresencePenalty = options.presencePenalty

            if effectiveRepeatPenalty != 1.0 || effectiveFrequencyPenalty != 0.0 || effectivePresencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPtr,
                    llama_sampler_init_penalties(
                        effectiveRepeatLastN,
                        effectiveRepeatPenalty,
                        effectiveFrequencyPenalty,
                        effectivePresencePenalty
                    )
                )
            }

            applySampling(sampler: samplerPtr, effectiveTemperature: effectiveTemperature, options: options)

            // Generate tokens one by one
            var generatedText = ""
            // Track position - for encoder-decoder models, we start from position 1 (after decoder start token)
            // For decoder-only models, we continue from the end of the prompt
            var n_cur: Int32 = hasEncoder ? 1 : batch.n_tokens

            for _ in 0 ..< maxTokens {
                // Sample next token from logits - llama_batch_get_one creates batch with single token at index 0
                let nextToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)
                llama_sampler_accept(sampler, nextToken)

                // Check for end of sequence
                if llama_vocab_is_eog(vocab, nextToken) {
                    break
                }

                // Convert token to text
                if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                    generatedText += tokenText
                }

                // Prepare batch for next token
                batch.n_tokens = 1
                batch.token[0] = nextToken
                batch.pos[0] = n_cur
                batch.n_seq_id[0] = 1
                if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                    seq_id[0] = 0
                }
                batch.logits[0] = 1

                n_cur += 1

                let decodeResult = llama_decode(context, batch)
                guard decodeResult == 0 else {
                    break
                }
            }

            return generatedText
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

        private func jsonSchemaString(for schema: GenerationSchema) throws -> String {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(schema)
            guard let jsonSchema = String(data: data, encoding: .utf8) else {
                throw LlamaLanguageModelError.schemaEncodingFailed
            }
            return jsonSchema
        }

        private func tokenizerInfo(
            for vocab: OpaquePointer,
            vocabSize: Int,
            stopTokens: Set<Int>
        ) throws -> TokenizerInfo {
            guard vocabSize > 0 else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            var encodedVocab: [String] = []
            encodedVocab.reserveCapacity(vocabSize)
            for tokenId in 0 ..< vocabSize {
                let token = llama_token(tokenId)
                encodedVocab.append(tokenToText(vocab: vocab, token: token) ?? "")
            }

            let stopTokenIDs = stopTokens.map { Int32($0) }
            return try TokenizerInfo(
                encodedVocab: encodedVocab,
                encoding: .byteFallback,
                stopTokenIDs: stopTokenIDs,
                addPrefixSpace: false
            )
        }

        // MARK: - Structured JSON Generation

        private func generateStructuredJSON(
            context: OpaquePointer,
            prompt: String,
            schema: GenerationSchema,
            maxTokens: Int,
            options: ResolvedGenerationOptions
        ) async throws -> String {
            guard let vocab = llama_model_get_vocab(model!) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
            guard !promptTokens.isEmpty else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
            defer { llama_batch_free(batch) }

            let hasEncoder = try prepareInitialBatch(
                batch: &batch,
                promptTokens: promptTokens,
                model: model!,
                vocab: vocab,
                context: context,
                batchSize: options.batchSize
            )

            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }
            let samplerPointer = UnsafeMutablePointer<llama_sampler>(sampler)

            if options.repeatPenalty != 1.0 || options.frequencyPenalty != 0.0 || options.presencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPointer,
                    llama_sampler_init_penalties(
                        options.repeatLastN,
                        options.repeatPenalty,
                        options.frequencyPenalty,
                        options.presencePenalty
                    )
                )
            }
            applySampling(sampler: samplerPointer, effectiveTemperature: options.temperature, options: options)

            let vocabSize = Int(llama_vocab_n_tokens(vocab))
            let initialPosition: Int32 = hasEncoder ? 1 : batch.n_tokens
            let jsonSchema = try jsonSchemaString(for: schema)
            let grammar = Grammar(jsonSchema: jsonSchema, formatting: .compact, strictMode: true)
            let eosToken = Int(llama_vocab_eos(vocab))
            let eotTokenValue = llama_vocab_eot(vocab)
            let endOfTurnToken = eotTokenValue != LLAMA_TOKEN_NULL ? Int(eotTokenValue) : eosToken
            let endTokens: Set<Int> = [eosToken, endOfTurnToken]

            let tokenizerInfo = try tokenizerInfo(
                for: vocab,
                vocabSize: vocabSize,
                stopTokens: endTokens
            )
            let matcher = try await grammar.matcher(
                for: tokenizerInfo,
                stopTokens: endTokens.map { Int32($0) },
                terminatesWithoutStopToken: true
            )
            var bitmask = Grammar.Matcher.TokenBitmask(vocabSize: vocabSize)

            return try withUnsafeMutablePointer(to: &batch) { batchPointer in
                var backend = LlamaTokenBackend(
                    context: context,
                    vocab: vocab,
                    vocabSize: vocabSize,
                    sampler: samplerPointer,
                    batch: batchPointer,
                    position: initialPosition,
                    maximumTokens: maxTokens,
                    endTokens: endTokens,
                    tokenToTextFn: { [self] token in self.tokenToText(vocab: vocab, token: llama_token(token)) }
                )

                var output = ""
                while backend.remainingTokens > 0 {
                    bitmask.reset()
                    let needsMask = matcher.fillNextTokenBitmask(&bitmask)
                    let token = try backend.sample(using: bitmask, applyMask: needsMask)
                    if backend.endTokens.contains(token) {
                        break
                    }
                    guard matcher.accept(Int32(token)) else {
                        throw LlamaLanguageModelError.grammarMismatch
                    }
                    if let tokenText = backend.tokenText(token) {
                        output += tokenText
                    }
                    try backend.decode(token)
                    if matcher.isTerminated { break }
                }
                return output
            }
        }

        private struct LlamaTokenBackend: TokenBackend {
            let context: OpaquePointer
            let vocab: OpaquePointer
            let vocabSize: Int
            let sampler: UnsafeMutablePointer<llama_sampler>
            let batch: UnsafeMutablePointer<llama_batch>
            let tokenToTextFn: (Int) -> String?
            let tokensExcludedFromRepetitionPenalty: Set<Int>
            let endTokens: Set<Int>

            var position: Int32
            var remainingTokens: Int
            let totalTokenBudget: Int
            let eosToken: Int

            init(
                context: OpaquePointer,
                vocab: OpaquePointer,
                vocabSize: Int,
                sampler: UnsafeMutablePointer<llama_sampler>,
                batch: UnsafeMutablePointer<llama_batch>,
                position: Int32,
                maximumTokens: Int,
                endTokens: Set<Int>? = nil,
                tokenToTextFn: @escaping (Int) -> String?
            ) {
                self.context = context
                self.vocab = vocab
                self.vocabSize = vocabSize
                self.sampler = sampler
                self.batch = batch
                self.position = position
                self.remainingTokens = maximumTokens
                self.totalTokenBudget = maximumTokens
                self.eosToken = Int(llama_vocab_eos(vocab))

                if let endTokens {
                    self.endTokens = endTokens
                } else {
                    let eotTokenValue = llama_vocab_eot(vocab)
                    let endOfTurnToken = eotTokenValue != LLAMA_TOKEN_NULL ? Int(eotTokenValue) : eosToken
                    self.endTokens = [self.eosToken, endOfTurnToken]
                }

                self.tokenToTextFn = tokenToTextFn
                self.tokensExcludedFromRepetitionPenalty = Self.buildTokensExcludedFromRepetitionPenalty(
                    vocabSize: vocabSize,
                    tokenToText: tokenToTextFn
                )
            }

            func isSpecialToken(_ token: Int) -> Bool {
                let attributes = llama_vocab_get_attr(vocab, llama_token(token))
                return (attributes.rawValue & LLAMA_TOKEN_ATTR_CONTROL.rawValue) != 0
            }

            private static func buildTokensExcludedFromRepetitionPenalty(
                vocabSize: Int,
                tokenToText: (Int) -> String?
            ) -> Set<Int> {
                let excludedTexts: Set<String> = ["{", "}", "[", "]", ",", ":", "\""]
                var excluded = Set<Int>()
                excluded.reserveCapacity(excludedTexts.count * 4)

                for token in 0 ..< vocabSize {
                    guard let text = tokenToText(token) else { continue }
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if excludedTexts.contains(trimmed) {
                        excluded.insert(token)
                    }
                }

                return excluded
            }

            func tokenize(_ text: String) throws -> [Int] {
                let utf8Count = text.utf8.count
                let capacity = Int32(max(utf8Count * 2, 8))
                let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(capacity))
                defer { tokens.deallocate() }

                let tokenCount = llama_tokenize(
                    vocab,
                    text,
                    Int32(utf8Count),
                    tokens,
                    capacity,
                    false,
                    false
                )
                guard tokenCount > 0 else { return [] }
                return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount))).map { Int($0) }
            }

            func tokenText(_ token: Int) -> String? {
                tokenToTextFn(token)
            }

            mutating func decode(_ token: Int) throws {
                let llamaToken = llama_token(token)

                batch.pointee.n_tokens = 1
                batch.pointee.token[0] = llamaToken
                batch.pointee.pos[0] = position
                batch.pointee.n_seq_id[0] = 1
                if let seqIds = batch.pointee.seq_id, let seqId = seqIds[0] {
                    seqId[0] = 0
                }
                batch.pointee.logits[0] = 1

                position += 1
                remainingTokens -= 1

                let decodeResult = llama_decode(context, batch.pointee)
                guard decodeResult == 0 else {
                    throw LlamaLanguageModelError.decodingFailed
                }

                if !tokensExcludedFromRepetitionPenalty.contains(Int(llamaToken)) {
                    llama_sampler_accept(sampler, llamaToken)
                }
            }

            mutating func sample(using bitmask: Grammar.Matcher.TokenBitmask, applyMask: Bool) throws -> Int {
                guard let logits = llama_get_logits(context) else {
                    return eosToken
                }

                if applyMask {
                    for tokenIndex in 0 ..< vocabSize where !bitmask.isTokenAllowed(tokenIndex) {
                        logits[tokenIndex] = -Float.infinity
                    }
                }

                let tokenIndex = batch.pointee.n_tokens - 1
                return Int(llama_sampler_sample(sampler, context, tokenIndex))
            }

            mutating func sample(from allowedTokens: Set<Int>) throws -> Int {
                guard let logits = llama_get_logits(context) else {
                    return eosToken
                }

                for tokenIndex in 0 ..< vocabSize {
                    if !allowedTokens.contains(tokenIndex) {
                        logits[tokenIndex] = -Float.infinity
                    }
                }

                let tokenIndex = batch.pointee.n_tokens - 1
                return Int(llama_sampler_sample(sampler, context, tokenIndex))
            }
        }

        private func generateTextStream(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: ResolvedGenerationOptions
        ) -> AsyncThrowingStream<String, Error> {
            return AsyncThrowingStream { continuation in
                self.performTextGeneration(
                    context: context,
                    model: model,
                    prompt: prompt,
                    maxTokens: maxTokens,
                    options: options,
                    continuation: continuation
                )
            }
        }

        private func performTextGeneration(
            context: OpaquePointer,
            model: OpaquePointer,
            prompt: String,
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            continuation: AsyncThrowingStream<String, Error>.Continuation
        ) {
            do {
                guard let vocab = llama_model_get_vocab(model) else {
                    continuation.finish(throwing: LlamaLanguageModelError.contextInitializationFailed)
                    return
                }

                // Tokenize the prompt
                let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
                guard !promptTokens.isEmpty else {
                    continuation.finish(throwing: LlamaLanguageModelError.tokenizationFailed)
                    return
                }

                // Initialize batch
                var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
                defer { llama_batch_free(batch) }

                let hasEncoder = try prepareInitialBatch(
                    batch: &batch,
                    promptTokens: promptTokens,
                    model: model,
                    vocab: vocab,
                    context: context,
                    batchSize: options.batchSize
                )

                // Initialize sampler chain with options
                guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                    throw LlamaLanguageModelError.decodingFailed
                }
                defer { llama_sampler_free(sampler) }
                let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

                let effectiveTemperature = Float(options.temperature)

                // Apply repeat/frequency/presence penalties from custom options
                let effectiveRepeatPenalty = options.repeatPenalty
                let effectiveRepeatLastN = options.repeatLastN
                let effectiveFrequencyPenalty = options.frequencyPenalty
                let effectivePresencePenalty = options.presencePenalty

                if effectiveRepeatPenalty != 1.0 || effectiveFrequencyPenalty != 0.0 || effectivePresencePenalty != 0.0
                {
                    llama_sampler_chain_add(
                        samplerPtr,
                        llama_sampler_init_penalties(
                            effectiveRepeatLastN,
                            effectiveRepeatPenalty,
                            effectiveFrequencyPenalty,
                            effectivePresencePenalty
                        )
                    )
                }

                // Check for mirostat sampling (takes precedence over standard sampling)
                applySampling(sampler: samplerPtr, effectiveTemperature: effectiveTemperature, options: options)

                // Generate tokens one by one
                // Track position - for encoder-decoder models, we start from position 1 (after decoder start token)
                // For decoder-only models, we continue from the end of the prompt
                var n_cur: Int32 = hasEncoder ? 1 : batch.n_tokens

                for _ in 0 ..< maxTokens {
                    // Sample next token from logits of the last token we just decoded
                    let nextToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)
                    llama_sampler_accept(sampler, nextToken)

                    // Check for end of sequence
                    if llama_vocab_is_eog(vocab, nextToken) {
                        break
                    }

                    // Convert token to text and yield it
                    if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                        continuation.yield(tokenText)
                    }

                    // Prepare batch for next token
                    batch.n_tokens = 1
                    batch.token[0] = nextToken
                    batch.pos[0] = n_cur
                    batch.n_seq_id[0] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                        seq_id[0] = 0
                    }
                    batch.logits[0] = 1

                    n_cur += 1

                    let decodeResult = llama_decode(context, batch)
                    guard decodeResult == 0 else {
                        break
                    }
                }

                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        // MARK: - Image Validation

        private func validateNoImageSegments(in session: LanguageModelSession) throws {
            // Check for image segments in the most recent prompt from the transcript
            for entry in session.transcript.reversed() {
                if case .prompt(let p) = entry {
                    for segment in p.segments {
                        if case .image = segment {
                            throw LlamaLanguageModelError.unsupportedFeature
                        }
                    }
                    break
                }
            }
        }

        // MARK: - Helper Methods

        /// Prepares the initial batch for text generation, handling encoder-decoder vs decoder-only models.
        ///
        /// - Parameters:
        ///   - batch: The batch to prepare (must be initialized with sufficient capacity).
        ///   - promptTokens: The tokenized prompt tokens.
        ///   - model: The loaded model.
        ///   - vocab: The model vocabulary.
        ///   - context: The model context.
        ///   - batchSize: The batch capacity to validate against (prevents buffer overflow).
        /// - Returns: `true` if the model has an encoder (for position tracking during generation).
        /// - Throws: `insufficientMemory` if prompt token count exceeds batch capacity, `encoderOnlyModel` if the model cannot generate text, `encodingFailed` or `decodingFailed` on failure.
        private func prepareInitialBatch(
            batch: inout llama_batch,
            promptTokens: [llama_token],
            model: OpaquePointer,
            vocab: OpaquePointer,
            context: OpaquePointer,
            batchSize: UInt32
        ) throws -> Bool {
            // Validate that prompt token count doesn't exceed batch capacity to prevent buffer overflow
            guard promptTokens.count <= batchSize else {
                throw LlamaLanguageModelError.insufficientMemory
            }

            let hasEncoder = llama_model_has_encoder(model)
            let hasDecoder = llama_model_has_decoder(model)

            if hasEncoder {
                // For encoder models, first encode the prompt
                batch.n_tokens = Int32(promptTokens.count)
                for i in 0 ..< promptTokens.count {
                    let idx = Int(i)
                    batch.token[idx] = promptTokens[idx]
                    batch.pos[idx] = Int32(i)
                    batch.n_seq_id[idx] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[idx] {
                        seq_id[0] = 0
                    }
                    batch.logits[idx] = 0
                }

                guard llama_encode(context, batch) == 0 else {
                    throw LlamaLanguageModelError.encodingFailed
                }

                if hasDecoder {
                    // For encoder-decoder models, start decoding with decoder start token
                    var decoderStartToken = llama_model_decoder_start_token(model)
                    if decoderStartToken == LLAMA_TOKEN_NULL {
                        decoderStartToken = llama_vocab_bos(vocab)
                    }

                    batch.n_tokens = 1
                    batch.token[0] = decoderStartToken
                    batch.pos[0] = 0
                    batch.n_seq_id[0] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                        seq_id[0] = 0
                    }
                    batch.logits[0] = 1

                    guard llama_decode(context, batch) == 0 else {
                        throw LlamaLanguageModelError.decodingFailed
                    }
                } else {
                    // Encoder-only model (like BERT) - cannot generate text.
                    // This architectural check complements the earlier KV cache check,
                    // catching models by their architecture type.
                    throw LlamaLanguageModelError.encoderOnlyModel
                }
            } else {
                // Standard decoder-only model (most LLMs)
                batch.n_tokens = Int32(promptTokens.count)
                for i in 0 ..< promptTokens.count {
                    let idx = Int(i)
                    batch.token[idx] = promptTokens[idx]
                    batch.pos[idx] = Int32(i)
                    batch.n_seq_id[idx] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[idx] {
                        seq_id[0] = 0
                    }
                    batch.logits[idx] = 0
                }

                if batch.n_tokens > 0 {
                    batch.logits[Int(batch.n_tokens) - 1] = 1
                }

                guard llama_decode(context, batch) == 0 else {
                    throw LlamaLanguageModelError.decodingFailed
                }
            }

            return hasEncoder
        }

        private func formatPrompt(
            for session: LanguageModelSession,
            extraSystemMessage: String? = nil
        ) throws -> String {
            guard let model = self.model else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            var messages: [(role: String, content: String)] = []

            for entry in session.transcript {
                switch entry {
                case .instructions(let instructions):
                    let text = extractText(from: instructions.segments)
                    if !text.isEmpty {
                        messages.append(("system", text))
                    }

                case .prompt(let prompt):
                    let text = extractText(from: prompt.segments)
                    if !text.isEmpty {
                        messages.append(("user", text))
                    }

                case .response(let response):
                    let text = extractText(from: response.segments)
                    if !text.isEmpty {
                        messages.append(("assistant", text))
                    }

                default:
                    break
                }
            }

            if let extraSystemMessage, !extraSystemMessage.isEmpty {
                messages.append(("system", extraSystemMessage))
            }

            // Keep C strings alive while using them
            let cRoles = messages.map { strdup($0.role) }
            let cContents = messages.map { strdup($0.content) }

            defer {
                cRoles.forEach { free($0) }
                cContents.forEach { free($0) }
            }

            var cMessages = [llama_chat_message]()
            for i in 0 ..< messages.count {
                cMessages.append(llama_chat_message(role: cRoles[i], content: cContents[i]))
            }

            // Get chat template embedded in the model's GGUF file (e.g., Llama 3, Mistral, ChatML)
            let tmpl = llama_model_chat_template(model, nil)

            // Get required buffer size
            let requiredSize = llama_chat_apply_template(
                tmpl,
                cMessages,
                cMessages.count,
                true,  // add_ass: Add assistant generation prompt
                nil,
                0
            )

            guard requiredSize > 0 else {
                throw LlamaLanguageModelError.encodingFailed
            }

            // Allocate buffer and apply template
            var buffer = [CChar](repeating: 0, count: Int(requiredSize) + 1)

            let result = llama_chat_apply_template(
                tmpl,
                cMessages,
                cMessages.count,
                true,
                &buffer,
                Int32(buffer.count)
            )

            guard result > 0 else {
                throw LlamaLanguageModelError.encodingFailed
            }

            return buffer.withUnsafeBytes { rawBuffer in
                String(decoding: rawBuffer.prefix(Int(result)), as: UTF8.self)
            }
        }

        private func extractText(from segments: [Transcript.Segment]) -> String {
            segments.compactMap { segment -> String? in
                if case .text(let t) = segment { return t.content }
                return nil
            }.joined()
        }

        private func tokenizeText(vocab: OpaquePointer, text: String) throws -> [llama_token] {
            let utf8Count = text.utf8.count
            let maxTokens = Int32(max(utf8Count * 2, 8))  // Rough estimate, minimum capacity
            let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(maxTokens))
            defer { tokens.deallocate() }

            let tokenCount = llama_tokenize(
                vocab,
                text,
                Int32(utf8Count),
                tokens,
                maxTokens,
                true,  // addSpecial
                true  // parseSpecial
            )

            guard tokenCount > 0 else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
        }

        private func tokenToText(vocab: OpaquePointer, token: llama_token) -> String? {
            // First attempt with a reasonable buffer
            var cap: Int32 = 64
            var buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(cap))
            defer { buf.deallocate() }

            var written = llama_token_to_piece(
                vocab,
                token,
                buf,
                cap,
                0,
                false
            )

            if written < 0 {
                // Reallocate to the required size and retry
                cap = -written
                buf.deallocate()
                buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(cap))
                written = llama_token_to_piece(
                    vocab,
                    token,
                    buf,
                    cap,
                    0,
                    false
                )
            }

            let count = Int(max(0, written))
            if count == 0 { return nil }

            // Create String from exact byte count (no reliance on NUL termination)
            let rawPtr = UnsafeRawPointer(buf)
            let u8Ptr = rawPtr.assumingMemoryBound(to: UInt8.self)
            let bytes = UnsafeBufferPointer(start: u8Ptr, count: count)
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    /// Errors that can occur when using LlamaLanguageModel
    public enum LlamaLanguageModelError: Error, LocalizedError {
        case modelLoadFailed
        case contextInitializationFailed
        case tokenizationFailed
        case encodingFailed
        case decodingFailed
        case invalidModelPath
        case insufficientMemory
        case unsupportedFeature
        case encoderOnlyModel
        case schemaEncodingFailed
        case grammarMismatch

        public var errorDescription: String? {
            switch self {
            case .modelLoadFailed:
                return "Failed to load model from file"
            case .contextInitializationFailed:
                return "Failed to initialize context"
            case .tokenizationFailed:
                return "Failed to tokenize input text"
            case .encodingFailed:
                return "Failed to encode prompt"
            case .decodingFailed:
                return "Failed to decode response"
            case .invalidModelPath:
                return "Invalid model file path"
            case .insufficientMemory:
                return "Insufficient memory for operation"
            case .unsupportedFeature:
                return "This LlamaLanguageModel does not support image segments"
            case .encoderOnlyModel:
                return "This model is encoder-only (e.g., BERT) and cannot generate text"
            case .schemaEncodingFailed:
                return "Failed to encode the JSON schema for structured generation"
            case .grammarMismatch:
                return "Grammar constraints could not be satisfied during generation"
            }
        }
    }
#endif  // Llama
