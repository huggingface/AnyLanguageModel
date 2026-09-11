import Foundation
#if Llama
    import JSONSchema
    import LlamaSwift

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

            /// Text appended after the assistant header of the rendered prompt.
            ///
            /// The model continues generating from this text.
            /// Use it to steer the start of the response,
            /// for example by prefilling an empty `<think></think>` block
            /// to suppress a model's default reasoning output
            /// when its chat template offers no switch for it.
            public var assistantPrefill: String?

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
                mirostat: MirostatMode? = nil,
                assistantPrefill: String? = nil
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
                self.assistantPrefill = assistantPrefill
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

        /// The number of model layers to offload to the GPU.
        ///
        /// A negative value offloads all layers, and `0` runs entirely on the CPU.
        public let gpuLayers: Int32

        /// The path to the multimodal projector GGUF file, when the model has one.
        ///
        /// Prompts may include image segments only when a projector is loaded.
        public let mmprojPath: String?

        /// The default GPU layer count for the current platform.
        ///
        /// All layers are offloaded by default: the prebuilt llama.cpp binaries
        /// ship with Metal enabled and the shader library embedded. The simulator
        /// defaults to CPU-only execution, which remains the reliable
        /// configuration there.
        public static var defaultGPULayerCount: Int32 {
            #if targetEnvironment(simulator)
                return 0
            #else
                return -1
            #endif
        }

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
            var assistantPrefill: String?
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
                assistantPrefill: String? = nil,
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
                self.assistantPrefill = assistantPrefill
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
                        assistantPrefill: base.assistantPrefill,
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
                self.assistantPrefill = options.assistantPrefill ?? base.assistantPrefill
                self.sampling = sampling ?? base.sampling
                self.maximumResponseTokens = maximumResponseTokens ?? base.maximumResponseTokens
            }
        }

        /// The loaded model instance
        private var model: OpaquePointer?

        /// The model's vocabulary
        private var vocab: OpaquePointer?

        /// The multimodal projector context, when a projector file was provided
        private var mtmdContext: OpaquePointer?
        /// `mtmd_helper_eval_chunks` is not thread-safe and every multimodal generation
        /// shares the one projector context, so bitmap init through eval runs under this lock.
        private let projectorLock = NSLock()

        private func projectorLocked<T>(_ body: () throws -> T) rethrows -> T {
            projectorLock.lock()
            defer { projectorLock.unlock() }
            return try body()
        }

        /// Serializes the loaded-state check, loading, and publication of the model,
        /// vocabulary, and projector so concurrent first requests cannot reload them.
        private let modelLoadLock = NSLock()

        /// Whether the model is currently loaded. Protected by `modelLoadLock`.
        private var isModelLoaded: Bool = false

        /// A context kept alive for one session so exchanges reuse its state.
        private struct CachedSessionContext {
            let sessionID: ObjectIdentifier
            let context: OpaquePointer
            var tokens: [llama_token]
            let contextSize: UInt32
            let batchSize: UInt32
            /// Whether a generation is currently decoding on the context.
            var isCheckedOut: Bool
            /// Whether the context should be freed once the current generation releases it.
            var discardWhenReleased: Bool
        }

        /// Guards `cachedSessionContext`. A generation checks the cached context out
        /// for its whole run, so a concurrent generation for another session never
        /// frees a context that is still decoding: it runs on a transient context
        /// instead and leaves the cache untouched.
        private let sessionContextLock = NSLock()
        private var cachedSessionContext: CachedSessionContext?

        /// The number of prompt tokens reused from the cached context by the most
        /// recent chat generation.
        internal private(set) var lastReusedTokenCount: Int = 0

        /// The number of prompt tokens decoded by the most recent chat generation.
        internal private(set) var lastPrefillTokenCount: Int = 0

        /// Frees the cached per-session context and the state it holds.
        ///
        /// The cached context, including its KV state, otherwise lives as long as the
        /// model. The next chat generation prefills its full prompt again. Call this
        /// under memory pressure or when a session is discarded. If a generation is
        /// running on the cached context, it is freed as soon as that generation ends.
        public func clearCachedContext() {
            discardCachedSessionContext()
        }

        private func discardCachedSessionContext() {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            guard var cached = cachedSessionContext else { return }
            if cached.isCheckedOut {
                cached.discardWhenReleased = true
                cachedSessionContext = cached
                return
            }
            llama_free(cached.context)
            cachedSessionContext = nil
        }

        private func recordCachedTokens(_ tokens: [llama_token], context: OpaquePointer) {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            guard var cached = cachedSessionContext, cached.context == context else { return }
            cached.tokens = tokens
            cachedSessionContext = cached
        }

        private func isCachedSessionContext(_ context: OpaquePointer) -> Bool {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            return cachedSessionContext?.context == context
        }

        /// Returns a context obtained from `acquireSessionContext`. The cached
        /// context is checked back in (or freed, if a discard was requested while
        /// it was busy); a transient context is freed.
        private func releaseSessionContext(_ context: OpaquePointer) {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            if var cached = cachedSessionContext, cached.context == context {
                if cached.discardWhenReleased {
                    llama_free(cached.context)
                    cachedSessionContext = nil
                } else {
                    cached.isCheckedOut = false
                    cachedSessionContext = cached
                }
                return
            }
            llama_free(context)
        }

        /// Returns a context for the session along with the index of the first
        /// prompt token that still needs to be decoded. Pair every call with
        /// `releaseSessionContext(_:)` once generation ends.
        ///
        /// A cached context whose recorded tokens share a prefix with the prompt
        /// keeps that prefix: matching state past the divergence point is removed
        /// with `llama_memory_seq_rm`, and backends whose state cannot be rewound
        /// (recurrent models) fall back to clearing the memory and decoding the
        /// full prompt. The final prompt token is always re-decoded so sampling
        /// has fresh logits.
        ///
        /// While another generation holds the cached context, the caller gets a
        /// transient context that decodes the full prompt and is not cached.
        private func acquireSessionContext(
            for session: LanguageModelSession,
            promptTokens: [llama_token],
            options: ResolvedGenerationOptions
        ) throws -> (context: OpaquePointer, startIndex: Int) {
            let sessionID = ObjectIdentifier(session)
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }

            if var cached = cachedSessionContext,
                !cached.isCheckedOut,
                cached.sessionID == sessionID,
                cached.contextSize == options.contextSize,
                cached.batchSize == options.batchSize
            {
                var common = 0
                while common < cached.tokens.count, common < promptTokens.count,
                    cached.tokens[common] == promptTokens[common]
                {
                    common += 1
                }
                if common == promptTokens.count {
                    common = max(0, promptTokens.count - 1)
                }
                if common < cached.tokens.count {
                    let memory = llama_get_memory(cached.context)
                    if !llama_memory_seq_rm(memory, 0, llama_pos(common), -1) {
                        llama_memory_clear(memory, true)
                        common = 0
                    }
                }
                cached.tokens = Array(promptTokens.prefix(common))
                cached.isCheckedOut = true
                cachedSessionContext = cached
                return (cached.context, common)
            }

            if let cached = cachedSessionContext, cached.isCheckedOut {
                return (try makeFreshContext(options: options), 0)
            }

            if let cached = cachedSessionContext {
                llama_free(cached.context)
                cachedSessionContext = nil
            }
            let contextParams = createContextParams(from: options)
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }
            guard llama_get_memory(context) != nil else {
                llama_free(context)
                throw LlamaLanguageModelError.encoderOnlyModel
            }
            cachedSessionContext = CachedSessionContext(
                sessionID: sessionID,
                context: context,
                tokens: [],
                contextSize: options.contextSize,
                batchSize: options.batchSize,
                isCheckedOut: true,
                discardWhenReleased: false
            )
            return (context, 0)
        }

        /// Creates a single-use context for generations that do not reuse state.
        private func makeFreshContext(options: ResolvedGenerationOptions) throws -> OpaquePointer {
            let contextParams = createContextParams(from: options)
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }
            guard llama_get_memory(context) != nil else {
                llama_free(context)
                throw LlamaLanguageModelError.encoderOnlyModel
            }
            llama_set_causal_attn(context, true)
            llama_set_n_threads(context, options.threads, options.threads)
            return context
        }

        /// Runs a chat text generation for the session, reusing the session's
        /// cached context when its state matches a prefix of the prompt.
        private func generateChatText(
            session: LanguageModelSession,
            prompt: String,
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            onToken: (String) -> Bool
        ) throws {
            guard let model = self.model, let vocab = llama_model_get_vocab(model) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
            guard !promptTokens.isEmpty else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            if llama_model_has_encoder(model) {
                let context = try makeFreshContext(options: options)
                defer { llama_free(context) }
                try performTokenGeneration(
                    context: context,
                    vocab: vocab,
                    promptTokens: promptTokens,
                    startIndex: 0,
                    maxTokens: maxTokens,
                    options: options,
                    onToken: onToken
                )
                return
            }

            let (context, startIndex) = try acquireSessionContext(
                for: session,
                promptTokens: promptTokens,
                options: options
            )
            defer { releaseSessionContext(context) }
            llama_set_causal_attn(context, true)
            llama_set_n_threads(context, options.threads, options.threads)

            do {
                try performTokenGeneration(
                    context: context,
                    vocab: vocab,
                    promptTokens: promptTokens,
                    startIndex: startIndex,
                    maxTokens: maxTokens,
                    options: options,
                    onToken: onToken
                )
            } catch {
                if isCachedSessionContext(context) {
                    discardCachedSessionContext()
                }
                throw error
            }
        }

        /// Creates a Llama language model.
        ///
        /// - Parameters:
        ///   - modelPath: The path to the GGUF model file.
        ///   - gpuLayers: The number of model layers to offload to the GPU.
        ///     Defaults to ``defaultGPULayerCount``.
        ///   - mmprojPath: The path to a multimodal projector GGUF file matching
        ///     the model. When provided, prompts may include image segments,
        ///     which are encoded through the projector. Defaults to `nil`
        ///     (text only).
        public init(
            modelPath: String,
            gpuLayers: Int32 = LlamaLanguageModel.defaultGPULayerCount,
            mmprojPath: String? = nil
        ) {
            self.modelPath = modelPath
            self.gpuLayers = gpuLayers
            self.mmprojPath = mmprojPath
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
            if let cached = cachedSessionContext {
                llama_free(cached.context)
            }
            if let mtmdContext = mtmdContext {
                mtmd_free(mtmdContext)
            }
            if let model = model {
                llama_model_free(model)
            }
        }

        // MARK: - Tool calling

        /// Prompt-side tool state for one exchange: the detected syntax, the
        /// session's tool definitions, and the tool turns produced so far in
        /// the current resolve-and-continue loop.
        struct LlamaToolPromptContext {
            let format: LlamaToolCallFormat
            let definitions: [LlamaToolDefinition]
            var pendingEntries: [Transcript.Entry] = []

            init(format: LlamaToolCallFormat, tools: [any Tool]) throws {
                self.format = format
                self.definitions = try tools.filter(\.includesSchemaInInstructions).map { tool in
                    let schema = tool.parameters.withResolvedRoot() ?? tool.parameters
                    let data = try JSONEncoder().encode(schema)
                    let parameters = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    return LlamaToolDefinition(
                        name: tool.name,
                        description: tool.description,
                        parameters: parameters
                    )
                }
            }
        }

        private struct ToolInvocationResult {
            let call: Transcript.ToolCall
            let output: Transcript.ToolOutput
        }

        private enum ToolResolutionOutcome {
            case stop(calls: [Transcript.ToolCall])
            case invocations([ToolInvocationResult])
        }

        private static func maxToolIterationsExceededError(limit: Int) -> LanguageModelSession.GenerationError {
            .decodingFailure(
                .init(
                    debugDescription:
                        "Exceeded maximum tool iterations (\(limit)) while processing Llama tool calls."
                )
            )
        }

        private static func repeatedToolCallLoopError() -> LanguageModelSession.GenerationError {
            .decodingFailure(
                .init(
                    debugDescription:
                        "Detected repeated Llama tool-call signature and aborted to avoid an infinite tool loop."
                )
            )
        }

        private func currentToolCallFormat() -> LlamaToolCallFormat {
            guard let model = self.model else { return .hermesJSON }
            let template = llama_model_chat_template(model, nil).map { String(cString: $0) }
            return LlamaToolCallFormat.detect(template: template)
        }

        private func makeToolPromptContext(for session: LanguageModelSession) throws -> LlamaToolPromptContext? {
            guard !session.tools.isEmpty, self.model != nil else { return nil }
            return try LlamaToolPromptContext(format: currentToolCallFormat(), tools: session.tools)
        }

        private func makeTranscriptToolCalls(
            from parsedCalls: [LlamaParsedToolCall]
        ) throws -> [Transcript.ToolCall] {
            try parsedCalls.map { parsed in
                Transcript.ToolCall(
                    id: UUID().uuidString,
                    toolName: parsed.name,
                    arguments: try GeneratedContent(json: parsed.argumentsJSON)
                )
            }
        }

        private func resolveToolCalls(
            _ parsedCalls: [LlamaParsedToolCall],
            session: LanguageModelSession
        ) async throws -> ToolResolutionOutcome {
            if parsedCalls.isEmpty { return .invocations([]) }

            var toolsByName: [String: any Tool] = [:]
            for tool in session.tools where toolsByName[tool.name] == nil {
                toolsByName[tool.name] = tool
            }

            let transcriptCalls = try makeTranscriptToolCalls(from: parsedCalls)

            if let delegate = session.toolExecutionDelegate {
                await delegate.didGenerateToolCalls(transcriptCalls, in: session)
            }

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

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            if mmprojPath == nil {
                try validateNoImageSegments(in: session)
            }
            try ensureModelLoaded()

            let runtimeOptions = resolvedOptions(from: options)
            let structuredOptions = resolvedStructuredOptions(from: options)

            let imageMarker = mtmdContext != nil ? String(cString: mtmd_default_marker()) : nil

            if type == String.self {
                let maxTokens = runtimeOptions.maximumResponseTokens ?? 100
                let outputFormat = currentToolCallFormat()
                var toolContext = try makeToolPromptContext(for: session)
                let maxToolIterations = 8
                var toolIteration = 0
                var previousToolCallSignature: String?
                var allEntries: [Transcript.Entry] = []
                var text = ""

                generationLoop: while true {
                    var promptImages: [Data] = []
                    let fullPrompt = try formatPrompt(
                        for: session,
                        extraSystemMessage: nil,
                        assistantPrefill: runtimeOptions.assistantPrefill,
                        imageMarker: imageMarker,
                        images: &promptImages,
                        toolContext: toolContext
                    )

                    var accumulated = ""
                    let terminator = toolContext?.format.callTerminator
                    let collectToken: (String) -> Bool = { tokenText in
                        accumulated += tokenText
                        if let terminator,
                            accumulated.suffix(terminator.count + 8).contains(terminator)
                        {
                            return false
                        }
                        return true
                    }

                    if promptImages.isEmpty {
                        try generateChatText(
                            session: session,
                            prompt: fullPrompt,
                            maxTokens: maxTokens,
                            options: runtimeOptions,
                            onToken: collectToken
                        )
                    } else {
                        discardCachedSessionContext()
                        let context = try makeFreshContext(options: runtimeOptions)
                        defer { llama_free(context) }
                        try performMultimodalGeneration(
                            context: context,
                            prompt: fullPrompt,
                            images: promptImages,
                            maxTokens: maxTokens,
                            options: runtimeOptions,
                            onToken: collectToken
                        )
                    }

                    guard let format = toolContext?.format else {
                        text = outputFormat.streamingVisibleText(
                            in: accumulated,
                            withholdToolCalls: false,
                            holdPartialMarkers: false
                        )
                        break generationLoop
                    }
                    let (visibleText, parsedCalls) = format.parseToolCalls(in: accumulated)
                    // Keep the text from every round, as the streaming path does,
                    // so a preamble before a tool call is not lost.
                    text += visibleText
                    if parsedCalls.isEmpty {
                        break generationLoop
                    }

                    toolIteration += 1
                    if toolIteration > maxToolIterations {
                        let unresolved = try makeTranscriptToolCalls(from: parsedCalls)
                        allEntries.append(.toolCalls(Transcript.ToolCalls(unresolved)))
                        throw Self.maxToolIterationsExceededError(limit: maxToolIterations)
                    }
                    let signature =
                        parsedCalls
                        .map { "\($0.name):\($0.argumentsJSON)" }
                        .joined(separator: "|")
                    if signature == previousToolCallSignature {
                        let unresolved = try makeTranscriptToolCalls(from: parsedCalls)
                        allEntries.append(.toolCalls(Transcript.ToolCalls(unresolved)))
                        throw Self.repeatedToolCallLoopError()
                    }
                    previousToolCallSignature = signature

                    let resolution = try await resolveToolCalls(parsedCalls, session: session)
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
                        guard !invocations.isEmpty else {
                            break generationLoop
                        }
                        let callsEntry = Transcript.Entry.toolCalls(
                            Transcript.ToolCalls(invocations.map(\.call))
                        )
                        allEntries.append(callsEntry)
                        toolContext?.pendingEntries.append(callsEntry)
                        for invocation in invocations {
                            let outputEntry = Transcript.Entry.toolOutput(invocation.output)
                            allEntries.append(outputEntry)
                            toolContext?.pendingEntries.append(outputEntry)
                        }
                    }
                }

                return LanguageModelSession.Response(
                    content: text as! Content,
                    rawContent: GeneratedContent(text),
                    transcriptEntries: ArraySlice(allEntries)
                )
            } else {
                var promptImages: [Data] = []
                let fullPrompt: String
                if includeSchemaInPrompt {
                    fullPrompt = try formatPrompt(
                        for: session,
                        extraSystemMessage: schemaPrompt(for: type.generationSchema),
                        assistantPrefill: runtimeOptions.assistantPrefill,
                        imageMarker: imageMarker,
                        images: &promptImages
                    )
                } else {
                    fullPrompt = try formatPrompt(
                        for: session,
                        extraSystemMessage: nil,
                        assistantPrefill: runtimeOptions.assistantPrefill,
                        imageMarker: imageMarker,
                        images: &promptImages
                    )
                }
                // Structured output does not go through the projector yet; refuse rather
                // than answer about an image the model never saw.
                guard promptImages.isEmpty else {
                    throw LlamaLanguageModelError.unsupportedFeature
                }
                let context = try makeFreshContext(options: runtimeOptions)
                defer { llama_free(context) }
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

            if mmprojPath == nil {
                do {
                    try validateNoImageSegments(in: session)
                } catch {
                    return LanguageModelSession.ResponseStream(
                        stream: AsyncThrowingStream { continuation in
                            continuation.finish(throwing: error)
                        }
                    )
                }
            }

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            try ensureModelLoaded()

                            let runtimeOptions = resolvedOptions(from: options)
                            let maxTokens = runtimeOptions.maximumResponseTokens ?? 100
                            let outputFormat = self.currentToolCallFormat()
                            var toolContext = try self.makeToolPromptContext(for: session)
                            let maxToolIterations = 8
                            var toolIteration = 0
                            var previousToolCallSignature: String?
                            var accumulatedEntries: [Transcript.Entry] = []
                            var emittedBase = ""
                            var lastYieldedText: String?
                            let imageMarker =
                                self.mtmdContext != nil ? String(cString: mtmd_default_marker()) : nil

                            func yieldSnapshot(_ text: String) {
                                lastYieldedText = text
                                let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                                    content: (text as! Content).asPartiallyGenerated(),
                                    rawContent: GeneratedContent(text),
                                    transcriptEntries: ArraySlice(accumulatedEntries)
                                )
                                continuation.yield(snapshot)
                            }

                            generationLoop: while true {
                                var promptImages: [Data] = []
                                let fullPrompt = try self.formatPrompt(
                                    for: session,
                                    extraSystemMessage: nil,
                                    assistantPrefill: runtimeOptions.assistantPrefill,
                                    imageMarker: imageMarker,
                                    images: &promptImages,
                                    toolContext: toolContext
                                )

                                var roundRaw = ""
                                let terminator = toolContext?.format.callTerminator
                                let withholdToolCalls = toolContext != nil
                                let collectToken: (String) -> Bool = { tokenText in
                                    roundRaw += tokenText
                                    let visible = outputFormat.streamingVisibleText(
                                        in: roundRaw,
                                        withholdToolCalls: withholdToolCalls
                                    )
                                    let total = emittedBase + visible
                                    if !total.isEmpty, total != lastYieldedText {
                                        yieldSnapshot(total)
                                    }
                                    if let terminator,
                                        roundRaw.suffix(terminator.count + 8).contains(terminator)
                                    {
                                        return false
                                    }
                                    return true
                                }

                                if promptImages.isEmpty {
                                    try self.generateChatText(
                                        session: session,
                                        prompt: fullPrompt,
                                        maxTokens: maxTokens,
                                        options: runtimeOptions,
                                        onToken: collectToken
                                    )
                                } else {
                                    self.discardCachedSessionContext()
                                    let context = try self.makeFreshContext(options: runtimeOptions)
                                    defer { llama_free(context) }
                                    try self.performMultimodalGeneration(
                                        context: context,
                                        prompt: fullPrompt,
                                        images: promptImages,
                                        maxTokens: maxTokens,
                                        options: runtimeOptions,
                                        onToken: collectToken
                                    )
                                }

                                if Task.isCancelled {
                                    break generationLoop
                                }

                                let roundVisible = outputFormat.streamingVisibleText(
                                    in: roundRaw,
                                    withholdToolCalls: withholdToolCalls,
                                    holdPartialMarkers: false
                                )

                                guard let format = toolContext?.format else {
                                    emittedBase += roundVisible
                                    break generationLoop
                                }
                                let (_, parsedCalls) = format.parseToolCalls(in: roundRaw)
                                if parsedCalls.isEmpty {
                                    emittedBase += roundVisible
                                    break generationLoop
                                }

                                toolIteration += 1
                                if toolIteration > maxToolIterations {
                                    let unresolved = try self.makeTranscriptToolCalls(from: parsedCalls)
                                    accumulatedEntries.append(.toolCalls(Transcript.ToolCalls(unresolved)))
                                    throw Self.maxToolIterationsExceededError(limit: maxToolIterations)
                                }
                                let signature =
                                    parsedCalls
                                    .map { "\($0.name):\($0.argumentsJSON)" }
                                    .joined(separator: "|")
                                if signature == previousToolCallSignature {
                                    let unresolved = try self.makeTranscriptToolCalls(from: parsedCalls)
                                    accumulatedEntries.append(.toolCalls(Transcript.ToolCalls(unresolved)))
                                    throw Self.repeatedToolCallLoopError()
                                }
                                previousToolCallSignature = signature

                                let resolution = try await self.resolveToolCalls(parsedCalls, session: session)
                                switch resolution {
                                case .stop(let calls):
                                    emittedBase += roundVisible
                                    if !calls.isEmpty {
                                        accumulatedEntries.append(.toolCalls(Transcript.ToolCalls(calls)))
                                        yieldSnapshot(emittedBase)
                                    }
                                    break generationLoop
                                case .invocations(let invocations):
                                    guard !invocations.isEmpty else {
                                        emittedBase += roundVisible
                                        break generationLoop
                                    }
                                    let callsEntry = Transcript.Entry.toolCalls(
                                        Transcript.ToolCalls(invocations.map(\.call))
                                    )
                                    accumulatedEntries.append(callsEntry)
                                    toolContext?.pendingEntries.append(callsEntry)
                                    for invocation in invocations {
                                        let outputEntry = Transcript.Entry.toolOutput(invocation.output)
                                        accumulatedEntries.append(outputEntry)
                                        toolContext?.pendingEntries.append(outputEntry)
                                    }
                                    emittedBase += roundVisible
                                    yieldSnapshot(emittedBase)
                                }
                            }

                            if emittedBase != lastYieldedText {
                                yieldSnapshot(emittedBase)
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

        private func ensureModelLoaded() throws {
            modelLoadLock.lock()
            defer { modelLoadLock.unlock() }

            guard !isModelLoaded else { return }

            // Check if model file exists
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw LlamaLanguageModelError.invalidModelPath
            }

            // Initialize backend lazily - must be done before loading model
            llama_backend_init()

            // Free any existing model before loading a new one
            discardCachedSessionContext()
            if let existingContext = mtmdContext {
                mtmd_free(existingContext)
                self.mtmdContext = nil
            }
            if let existingModel = model {
                llama_model_free(existingModel)
                self.model = nil
            }

            let modelParams = createModelParams()
            guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            if let mmprojPath {
                guard FileManager.default.fileExists(atPath: mmprojPath) else {
                    llama_model_free(loadedModel)
                    throw LlamaLanguageModelError.invalidModelPath
                }
                var mtmdParams = mtmd_context_params_default()
                mtmdParams.use_gpu = gpuLayers != 0
                mtmdParams.print_timings = false
                mtmdParams.n_threads = legacyDefaults.threads
                guard let projector = mtmd_init_from_file(mmprojPath, loadedModel, mtmdParams) else {
                    llama_model_free(loadedModel)
                    throw LlamaLanguageModelError.modelLoadFailed
                }
                self.mtmdContext = projector
            }

            self.model = loadedModel
            self.vocab = llama_model_get_vocab(loadedModel)
            self.isModelLoaded = true
        }

        private func createModelParams() -> llama_model_params {
            var params = llama_model_default_params()
            params.n_gpu_layers = gpuLayers

            // Try to reduce memory usage
            params.load_mode = LLAMA_LOAD_MODE_MMAP
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

            let batchPointer = UnsafeMutablePointer<llama_batch>.allocate(capacity: 1)
            batchPointer.initialize(to: llama_batch_init(Int32(options.batchSize), 0, 1))
            defer {
                llama_batch_free(batchPointer.pointee)
                batchPointer.deinitialize(count: 1)
                batchPointer.deallocate()
            }

            let hasEncoder = try prepareInitialBatch(
                batch: &batchPointer.pointee,
                promptTokens: promptTokens,
                model: model!,
                vocab: vocab,
                context: context,
                batchSize: options.batchSize,
                contextSize: options.contextSize
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
                        llama_vocab_n_tokens(vocab),
                        options.repeatLastN,
                        options.repeatPenalty,
                        options.frequencyPenalty,
                        options.presencePenalty
                    )
                )
            }
            applySampling(sampler: samplerPointer, effectiveTemperature: options.temperature, options: options)

            let vocabSize = Int(llama_vocab_n_tokens(vocab))
            let initialPosition: Int32 = hasEncoder ? 1 : Int32(promptTokens.count)

            let backend = LlamaTokenBackend(
                context: context,
                vocab: vocab,
                vocabSize: vocabSize,
                sampler: samplerPointer,
                batch: batchPointer,
                position: initialPosition,
                maximumTokens: maxTokens,
                endTokens: [],
                tokenToTextFn: { [self] token in self.tokenToText(vocab: vocab, token: llama_token(token)) }
            )
            var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
            return try await generator.generate()
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

            mutating func decode(_ token: Int) async throws {
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

            mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
                // Masking llama_get_logits in place does not constrain
                // llama_sampler_sample: the context can refresh that buffer when
                // the sampler fetches logits, dropping the mask. Build a candidate
                // array holding only the allowed tokens and run the sampler chain
                // over it instead.
                guard let logits = llama_get_logits_ith(context, batch.pointee.n_tokens - 1) else {
                    return eosToken
                }

                var candidates = allowedTokens.compactMap { token -> llama_token_data? in
                    guard token >= 0, token < vocabSize else { return nil }
                    return llama_token_data(id: llama_token(token), logit: logits[token], p: 0)
                }
                guard !candidates.isEmpty else {
                    return eosToken
                }

                let selectedToken = candidates.withUnsafeMutableBufferPointer { buffer -> Int in
                    var array = llama_token_data_array(
                        data: buffer.baseAddress,
                        size: buffer.count,
                        selected: -1,
                        sorted: false
                    )
                    llama_sampler_apply(sampler, &array)
                    guard array.selected >= 0, array.selected < Int64(array.size), let data = array.data else {
                        return eosToken
                    }
                    return Int(data[Int(array.selected)].id)
                }
                return selectedToken
            }
        }

        private func performTokenGeneration(
            context: OpaquePointer,
            vocab: OpaquePointer,
            promptTokens: [llama_token],
            startIndex: Int,
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            onToken: (String) -> Bool
        ) throws {
            guard let model = self.model else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            sessionContextLock.lock()
            lastReusedTokenCount = startIndex
            lastPrefillTokenCount = promptTokens.count - startIndex
            sessionContextLock.unlock()

            // Initialize batch
            var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
            defer { llama_batch_free(batch) }

            let hasEncoder = try prepareInitialBatch(
                batch: &batch,
                promptTokens: promptTokens,
                model: model,
                vocab: vocab,
                context: context,
                batchSize: options.batchSize,
                contextSize: options.contextSize,
                startIndex: startIndex
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
                        llama_vocab_n_tokens(vocab),
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
            var n_cur: Int32 = hasEncoder ? 1 : Int32(promptTokens.count)
            var decodedTokens = promptTokens

            for _ in 0 ..< maxTokens {
                if Task.isCancelled {
                    break
                }

                // Sample next token from logits of the last token we just decoded
                let nextToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)
                llama_sampler_accept(sampler, nextToken)

                // Check for end of sequence
                if llama_vocab_is_eog(vocab, nextToken) {
                    break
                }

                // Convert token to text and yield it
                if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                    guard onToken(tokenText) else {
                        break
                    }
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
                decodedTokens.append(nextToken)
            }

            recordCachedTokens(decodedTokens, context: context)
        }

        /// Evaluates a marker-annotated multimodal prompt through the projector,
        /// then generates text tokens from the resulting state.
        private func performMultimodalGeneration(
            context: OpaquePointer,
            prompt: String,
            images: [Data],
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            onToken: (String) -> Bool
        ) throws {
            guard let mtmdContext, let model = self.model,
                let vocab = llama_model_get_vocab(model)
            else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            var pastPosition: llama_pos = 0
            try projectorLocked {
                var bitmaps: [OpaquePointer?] = []
                defer {
                    for bitmap in bitmaps {
                        if let bitmap {
                            mtmd_bitmap_free(bitmap)
                        }
                    }
                }
                for imageData in images {
                    // Pinned to the current llama.swift signature. llama.cpp master adds a
                    // trailing options argument to this helper; update alongside the dependency.
                    let wrapper = imageData.withUnsafeBytes { raw -> mtmd_helper_bitmap_wrapper in
                        mtmd_helper_bitmap_init_from_buf(
                            mtmdContext,
                            raw.bindMemory(to: UInt8.self).baseAddress,
                            imageData.count,
                            false
                        )
                    }
                    if let videoContext = wrapper.video_ctx {
                        if let bitmap = wrapper.bitmap {
                            mtmd_bitmap_free(bitmap)
                        }
                        mtmd_helper_video_free(videoContext)
                        throw LlamaLanguageModelError.unsupportedFeature
                    }
                    guard let bitmap = wrapper.bitmap else {
                        throw LlamaLanguageModelError.encodingFailed
                    }
                    bitmaps.append(bitmap)
                }

                guard let chunks = mtmd_input_chunks_init() else {
                    throw LlamaLanguageModelError.encodingFailed
                }
                defer { mtmd_input_chunks_free(chunks) }

                let tokenizeResult = prompt.withCString { cPrompt -> Int32 in
                    var inputText = mtmd_input_text(
                        text: cPrompt,
                        text_len: strlen(cPrompt),
                        add_special: true,
                        parse_special: true
                    )
                    return bitmaps.withUnsafeMutableBufferPointer { buffer in
                        mtmd_tokenize(mtmdContext, chunks, &inputText, buffer.baseAddress, buffer.count)
                    }
                }
                guard tokenizeResult == 0 else {
                    throw LlamaLanguageModelError.tokenizationFailed
                }

                let evalResult = mtmd_helper_eval_chunks(
                    mtmdContext,
                    context,
                    chunks,
                    0,
                    0,
                    Int32(options.batchSize),
                    true,
                    &pastPosition
                )
                guard evalResult == 0 else {
                    throw LlamaLanguageModelError.decodingFailed
                }
            }

            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }
            let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

            if options.repeatPenalty != 1.0 || options.frequencyPenalty != 0.0 || options.presencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPtr,
                    llama_sampler_init_penalties(
                        llama_vocab_n_tokens(vocab),
                        options.repeatLastN,
                        options.repeatPenalty,
                        options.frequencyPenalty,
                        options.presencePenalty
                    )
                )
            }
            applySampling(sampler: samplerPtr, effectiveTemperature: options.temperature, options: options)

            var batch = llama_batch_init(1, 0, 1)
            defer { llama_batch_free(batch) }

            var n_cur: Int32 = Int32(pastPosition)
            var sampleIndex: Int32 = -1

            for _ in 0 ..< maxTokens {
                if Task.isCancelled {
                    break
                }

                let nextToken = llama_sampler_sample(samplerPtr, context, sampleIndex)
                llama_sampler_accept(samplerPtr, nextToken)

                if llama_vocab_is_eog(vocab, nextToken) {
                    break
                }

                if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                    guard onToken(tokenText) else {
                        break
                    }
                }

                batch.n_tokens = 1
                batch.token[0] = nextToken
                batch.pos[0] = n_cur
                batch.n_seq_id[0] = 1
                if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                    seq_id[0] = 0
                }
                batch.logits[0] = 1

                n_cur += 1

                guard llama_decode(context, batch) == 0 else {
                    break
                }
                sampleIndex = 0
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
        /// Decoder-only prompts longer than the batch capacity are ingested in
        /// batch-sized chunks. Encoder models must fit the prompt in one batch.
        ///
        /// - Parameters:
        ///   - batch: The batch to prepare (must be initialized with sufficient capacity).
        ///   - promptTokens: The tokenized prompt tokens.
        ///   - model: The loaded model.
        ///   - vocab: The model vocabulary.
        ///   - context: The model context.
        ///   - batchSize: The batch capacity per decode call.
        ///   - contextSize: The context window the prompt must fit within.
        ///   - startIndex: The index of the first prompt token to decode. Earlier
        ///     tokens are already present in the context's state. Defaults to `0`.
        /// - Returns: `true` if the model has an encoder (for position tracking during generation).
        /// - Throws: `promptExceedsContextWindow` if the prompt cannot fit in the context window,
        ///   `insufficientMemory` if an encoder prompt exceeds the batch capacity, `encoderOnlyModel`
        ///   if the model cannot generate text, `encodingFailed` or `decodingFailed` on failure.
        private func prepareInitialBatch(
            batch: inout llama_batch,
            promptTokens: [llama_token],
            model: OpaquePointer,
            vocab: OpaquePointer,
            context: OpaquePointer,
            batchSize: UInt32,
            contextSize: UInt32,
            startIndex: Int = 0
        ) throws -> Bool {
            // Leave at least one context cell free for generation.
            guard promptTokens.count < contextSize else {
                throw LlamaLanguageModelError.promptExceedsContextWindow
            }

            let hasEncoder = llama_model_has_encoder(model)
            let hasDecoder = llama_model_has_decoder(model)

            // Encoder models ingest the full prompt in a single llama_encode call.
            guard !hasEncoder || (startIndex == 0 && promptTokens.count <= batchSize) else {
                throw LlamaLanguageModelError.insufficientMemory
            }

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
                // Standard decoder-only model (most LLMs): feed the prompt in
                // batch-sized chunks with absolute positions, requesting logits
                // only for the final token.
                let capacity = Int(batchSize)
                var start = startIndex
                while start < promptTokens.count {
                    let count = min(capacity, promptTokens.count - start)
                    batch.n_tokens = Int32(count)
                    for i in 0 ..< count {
                        batch.token[i] = promptTokens[start + i]
                        batch.pos[i] = Int32(start + i)
                        batch.n_seq_id[i] = 1
                        if let seq_ids = batch.seq_id, let seq_id = seq_ids[i] {
                            seq_id[0] = 0
                        }
                        batch.logits[i] = 0
                    }

                    if start + count == promptTokens.count {
                        batch.logits[count - 1] = 1
                    }

                    guard llama_decode(context, batch) == 0 else {
                        throw LlamaLanguageModelError.decodingFailed
                    }
                    start += count
                }
            }

            return hasEncoder
        }

        private func formatPrompt(
            for session: LanguageModelSession,
            extraSystemMessage: String? = nil,
            assistantPrefill: String? = nil,
            toolContext: LlamaToolPromptContext? = nil
        ) throws -> String {
            var images: [Data] = []
            return try formatPrompt(
                for: session,
                extraSystemMessage: extraSystemMessage,
                assistantPrefill: assistantPrefill,
                imageMarker: nil,
                images: &images,
                toolContext: toolContext
            )
        }

        private func formatPrompt(
            for session: LanguageModelSession,
            extraSystemMessage: String?,
            assistantPrefill: String?,
            imageMarker: String?,
            images: inout [Data],
            toolContext: LlamaToolPromptContext? = nil
        ) throws -> String {
            guard let model = self.model else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            var messages: [(role: String, content: String)] = []

            func appendEntry(_ entry: Transcript.Entry) throws {
                switch entry {
                case .instructions(let instructions):
                    let text = try extractContent(
                        from: instructions.segments,
                        imageMarker: imageMarker,
                        images: &images
                    )
                    if !text.isEmpty {
                        messages.append(("system", text))
                    }

                case .prompt(let prompt):
                    let text = try extractContent(
                        from: prompt.segments,
                        imageMarker: imageMarker,
                        images: &images
                    )
                    if !text.isEmpty {
                        messages.append(("user", text))
                    }

                case .response(let response):
                    let text = try extractContent(
                        from: response.segments,
                        imageMarker: imageMarker,
                        images: &images
                    )
                    if !text.isEmpty {
                        messages.append(("assistant", text))
                    }

                case .toolCalls(let toolCalls):
                    guard let toolContext else { break }
                    let parsed = toolCalls.map {
                        LlamaParsedToolCall(name: $0.toolName, argumentsJSON: $0.arguments.jsonString)
                    }
                    if let last = messages.last, last.role == "assistant" {
                        let markup = toolContext.format.assistantText(for: parsed, precededByContent: true)
                        messages[messages.count - 1].content += markup
                    } else {
                        let markup = toolContext.format.assistantText(for: parsed, precededByContent: false)
                        messages.append(("assistant", markup))
                    }

                case .toolOutput(let output):
                    guard let toolContext else { break }
                    let message = toolContext.format.toolResponseMessage(
                        toolName: output.toolName,
                        segments: output.segments
                    )
                    if let last = messages.last, last.role == message.role, last.role == "user",
                        last.content.hasSuffix("</tool_response>")
                    {
                        messages[messages.count - 1].content += "\n" + message.content
                    } else {
                        messages.append(message)
                    }
                }
            }

            for entry in session.transcript {
                try appendEntry(entry)
            }
            if let toolContext {
                for entry in toolContext.pendingEntries {
                    try appendEntry(entry)
                }
            }

            if let toolContext, !toolContext.definitions.isEmpty {
                if let systemIndex = messages.firstIndex(where: { $0.role == "system" }) {
                    messages[systemIndex].content = try toolContext.format.systemMessage(
                        existingText: messages[systemIndex].content,
                        tools: toolContext.definitions
                    )
                } else {
                    let systemText = try toolContext.format.systemMessage(
                        existingText: "",
                        tools: toolContext.definitions
                    )
                    messages.insert(("system", systemText), at: 0)
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
                if let tmpl, String(cString: tmpl).contains("<|turn>") {
                    return Self.renderGemma4Prompt(messages: messages, assistantPrefill: assistantPrefill)
                }
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

            let rendered = buffer.withUnsafeBytes { rawBuffer in
                String(decoding: rawBuffer.prefix(Int(result)), as: UTF8.self)
            }

            if let assistantPrefill, !assistantPrefill.isEmpty {
                return rendered + assistantPrefill
            }
            return rendered
        }

        /// Renders the Gemma 4 canonical chat format, which
        /// `llama_chat_apply_template` does not recognize: turns open with
        /// `<|turn>role`, close with `<turn|>`, and the assistant role is named
        /// `model`. The BOS token is applied during tokenization.
        static func renderGemma4Prompt(
            messages: [(role: String, content: String)],
            assistantPrefill: String?
        ) -> String {
            var rendered = ""
            var openModelTurn = false
            for (index, message) in messages.enumerated() {
                if message.role == "tool" {
                    rendered += message.content
                    continue
                }
                let role = message.role == "assistant" ? "model" : message.role
                let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if role == "model" && openModelTurn {
                    rendered += content
                } else {
                    if openModelTurn {
                        rendered += "<turn|>\n"
                        openModelTurn = false
                    }
                    rendered += "<|turn>\(role)\n\(content)"
                }
                if role == "model" {
                    let nextRole = index + 1 < messages.count ? messages[index + 1].role : nil
                    if nextRole == "tool" || nextRole == "assistant" {
                        openModelTurn = true
                    } else {
                        rendered += "<turn|>\n"
                        openModelTurn = false
                    }
                } else {
                    rendered += "<turn|>\n"
                }
            }
            if !openModelTurn {
                rendered += "<|turn>model\n"
            }
            if let assistantPrefill, !assistantPrefill.isEmpty {
                rendered += assistantPrefill
            }
            return rendered
        }

        private func extractText(from segments: [Transcript.Segment]) -> String {
            segments.compactMap { segment -> String? in
                if case .text(let t) = segment { return t.content }
                return nil
            }.joined()
        }

        /// Extracts message content from segments, replacing each image segment
        /// with `imageMarker` and collecting its payload in order. Image segments
        /// throw ``LlamaLanguageModelError/unsupportedFeature`` when no marker is
        /// provided.
        private func extractContent(
            from segments: [Transcript.Segment],
            imageMarker: String?,
            images: inout [Data]
        ) throws -> String {
            var parts: [String] = []
            for segment in segments {
                switch segment {
                case .text(let t):
                    parts.append(t.content)
                case .image(let image):
                    guard let imageMarker else {
                        throw LlamaLanguageModelError.unsupportedFeature
                    }
                    switch image.source {
                    case .data(let data, _):
                        images.append(data)
                        parts.append(imageMarker)
                    case .url(let url):
                        guard url.isFileURL, let data = try? Data(contentsOf: url) else {
                            throw LlamaLanguageModelError.unsupportedFeature
                        }
                        images.append(data)
                        parts.append(imageMarker)
                    }
                default:
                    break
                }
            }
            return parts.joined()
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
        case promptExceedsContextWindow
        case unsupportedFeature
        case encoderOnlyModel

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
            case .promptExceedsContextWindow:
                return "Prompt is longer than the model's context window"
            case .unsupportedFeature:
                return "This LlamaLanguageModel does not support image segments"
            case .encoderOnlyModel:
                return "This model is encoder-only (e.g., BERT) and cannot generate text"
            }
        }
    }
#endif  // Llama
