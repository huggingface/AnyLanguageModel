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

        /// Configures MLX-specific generation behavior.
        ///
        /// Set these values through ``GenerationOptions`` using
        /// `GenerationOptions[custom: MLXLanguageModel.self]`.
        public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions, Codable {
            /// Limits how many tokens the KV cache retains.
            ///
            /// Set this to `nil` to use the backend default.
            public var maxKVSize: Int?
            /// Sets the KV-cache quantization bit width.
            ///
            /// Set this to `nil` to disable KV quantization.
            public var kvBits: Int?
            /// Sets the token group size used for KV quantization.
            public var kvGroupSize: Int
            /// Sets the token offset where quantized KV storage starts.
            public var quantizedKVStart: Int

            /// Creates MLX-specific generation options.
            ///
            /// - Parameters:
            ///   - maxKVSize: The maximum number of tokens to retain in KV cache storage.
            ///     Pass `nil` to use the backend default.
            ///   - kvBits: The KV-cache quantization bit width.
            ///     Pass `nil` to disable KV quantization.
            ///   - kvGroupSize: The token group size used for KV quantization.
            ///   - quantizedKVStart: The token index where quantized KV storage begins.
            public init(
                maxKVSize: Int? = nil,
                kvBits: Int? = nil,
                kvGroupSize: Int = 64,
                quantizedKVStart: Int = 0
            ) {
                self.maxKVSize = maxKVSize
                self.kvBits = kvBits
                self.kvGroupSize = kvGroupSize
                self.quantizedKVStart = quantizedKVStart
            }
        }

        /// Controls GPU buffer-pool limits during active and idle phases.
        public struct GPUMemoryConfiguration: Sendable, Hashable {
            /// The cache limit applied while at least one generation is active.
            public var activeCacheLimit: Int
            /// The cache limit applied when no generations are active.
            public var idleCacheLimit: Int
            /// Indicates whether MLX clears cached GPU buffers on safe eviction.
            public var clearCacheOnEviction: Bool

            /// Creates a GPU-memory configuration for MLX generations.
            ///
            /// - Parameters:
            ///   - activeCacheLimit: The GPU cache-limit value used during active generation.
            ///   - idleCacheLimit: The GPU cache-limit value used while idle.
            ///   - clearCacheOnEviction: A Boolean value that indicates whether to clear
            ///     cached GPU buffers when eviction is safe.
            public init(
                activeCacheLimit: Int,
                idleCacheLimit: Int,
                clearCacheOnEviction: Bool = true
            ) {
                self.activeCacheLimit = activeCacheLimit
                self.idleCacheLimit = idleCacheLimit
                self.clearCacheOnEviction = clearCacheOnEviction
            }

            /// Returns a memory configuration using physical-memory heuristics.
            ///
            /// The active limit scales with device RAM,
            /// and the idle limit stays conservative to reduce background memory pressure.
            public static var automatic: GPUMemoryConfiguration {
                let ramBytes = ProcessInfo.processInfo.physicalMemory
                let ramGB = ramBytes / (1024 * 1024 * 1024)
                let active: Int
                switch ramGB {
                case ..<4:
                    active = 128_000_000
                case ..<6:
                    active = 256_000_000
                case ..<8:
                    active = 512_000_000
                default:
                    active = 768_000_000
                }

                return .init(
                    activeCacheLimit: active,
                    idleCacheLimit: 50_000_000,
                    clearCacheOnEviction: true
                )
            }

            /// Returns a memory configuration that leaves GPU cache effectively unconstrained.
            ///
            /// Use this when your application prefers maximum reuse over memory reclamation.
            public static var unconstrained: GPUMemoryConfiguration {
                .init(
                    activeCacheLimit: Int.max,
                    idleCacheLimit: Int.max,
                    clearCacheOnEviction: false
                )
            }
        }

        private struct CacheConfigSignature: Equatable {
            let maxKVSize: Int?
            let kvBits: Int?
            let kvGroupSize: Int
            let quantizedKVStart: Int
        }

        private final class SessionCacheEntry: @unchecked Sendable {
            let kvCache: [MLXLMCommon.KVCache]
            let prefillTokenCount: Int
            let prefixTokens: [Int32]
            let cacheConfigSignature: CacheConfigSignature

            init(
                kvCache: [MLXLMCommon.KVCache],
                prefillTokenCount: Int,
                prefixTokens: [Int32],
                cacheConfigSignature: CacheConfigSignature
            ) {
                self.kvCache = kvCache
                self.prefillTokenCount = prefillTokenCount
                self.prefixTokens = prefixTokens
                self.cacheConfigSignature = cacheConfigSignature
            }
        }

        private final class SessionKVStore: @unchecked Sendable {
            private final class WeakSessionReference: @unchecked Sendable {
                weak var session: LanguageModelSession?

                init(_ session: LanguageModelSession) {
                    self.session = session
                }
            }

            private struct SessionBucket {
                let sessionReference: WeakSessionReference
                var modelEntries: [String: SessionCacheEntry]
            }

            private let lock = NSLock()
            private var buckets: [ObjectIdentifier: SessionBucket] = [:]

            func entry(
                for session: LanguageModelSession,
                modelKey: String
            ) -> SessionCacheEntry? {
                lock.withLock {
                    reapDeadSessionsLocked()
                    return buckets[ObjectIdentifier(session)]?.modelEntries[modelKey]
                }
            }

            func set(
                _ entry: SessionCacheEntry,
                for session: LanguageModelSession,
                modelKey: String
            ) {
                lock.withLock {
                    reapDeadSessionsLocked()
                    let id = ObjectIdentifier(session)
                    var bucket =
                        buckets[id]
                        ?? SessionBucket(
                            sessionReference: WeakSessionReference(session),
                            modelEntries: [:]
                        )
                    bucket.modelEntries[modelKey] = entry
                    buckets[id] = bucket
                }
            }

            func removeEntry(
                for session: LanguageModelSession,
                modelKey: String
            ) {
                lock.withLock {
                    reapDeadSessionsLocked()
                    let id = ObjectIdentifier(session)
                    guard var bucket = buckets[id] else {
                        return
                    }
                    bucket.modelEntries[modelKey] = nil
                    if bucket.modelEntries.isEmpty {
                        buckets[id] = nil
                    } else {
                        buckets[id] = bucket
                    }
                }
            }

            func removeEntries(forModelKey modelKey: String) {
                lock.withLock {
                    reapDeadSessionsLocked()
                    for id in Array(buckets.keys) {
                        guard var bucket = buckets[id] else {
                            continue
                        }
                        bucket.modelEntries[modelKey] = nil
                        if bucket.modelEntries.isEmpty {
                            buckets[id] = nil
                        } else {
                            buckets[id] = bucket
                        }
                    }
                }
            }

            func removeAll() {
                lock.withLock {
                    buckets.removeAll()
                }
            }

            private func reapDeadSessionsLocked() {
                let deadSessionIDs = buckets.compactMap { id, bucket in
                    bucket.sessionReference.session == nil ? id : nil
                }
                for id in deadSessionIDs {
                    buckets[id] = nil
                }
            }
        }

        private final class SessionGenerationGate: @unchecked Sendable {
            private let lock = NSLock()
            private var activeSessions: Set<ObjectIdentifier> = []

            func acquire(session: LanguageModelSession) -> Bool {
                lock.withLock {
                    let id = ObjectIdentifier(session)
                    guard !activeSessions.contains(id) else {
                        return false
                    }
                    activeSessions.insert(id)
                    return true
                }
            }

            func release(session: LanguageModelSession) {
                _ = lock.withLock {
                    activeSessions.remove(ObjectIdentifier(session))
                }
            }
        }

        private final class GPUMemoryManager: @unchecked Sendable {
            static let shared = GPUMemoryManager()

            private let lock = NSLock()
            private var knownConfigs: Set<GPUMemoryConfiguration> = []
            private var activeScopes: [UUID: GPUMemoryConfiguration] = [:]

            private init() {
                GPU.set(cacheLimit: GPUMemoryConfiguration.automatic.idleCacheLimit)
            }

            func register(_ configuration: GPUMemoryConfiguration) {
                var cacheLimitToSet: Int?
                lock.withLock {
                    knownConfigs.insert(configuration)
                    if activeScopes.isEmpty {
                        cacheLimitToSet = effectiveIdleLimit()
                    }
                }
                if let cacheLimitToSet {
                    GPU.set(cacheLimit: cacheLimitToSet)
                }
            }

            func markActive(_ configuration: GPUMemoryConfiguration) -> UUID {
                let id = UUID()
                let cacheLimitToSet = lock.withLock {
                    knownConfigs.insert(configuration)
                    activeScopes[id] = configuration
                    return effectiveActiveLimit()
                }
                GPU.set(cacheLimit: cacheLimitToSet)
                return id
            }

            func markIdle(scope id: UUID) {
                let cacheLimitToSet = lock.withLock {
                    activeScopes.removeValue(forKey: id)
                    if activeScopes.isEmpty {
                        return effectiveIdleLimit()
                    }
                    return effectiveActiveLimit()
                }
                GPU.set(cacheLimit: cacheLimitToSet)
            }

            func evictIfSafe() {
                var shouldUpdateCacheLimit = false
                var cacheLimitToSet = 0
                var shouldClearCache = false
                lock.withLock {
                    guard activeScopes.isEmpty else { return }
                    shouldUpdateCacheLimit = true
                    cacheLimitToSet = effectiveIdleLimit()
                    shouldClearCache = shouldClearOnEviction()
                }
                guard shouldUpdateCacheLimit else { return }
                GPU.set(cacheLimit: cacheLimitToSet)
                if shouldClearCache {
                    GPU.clearCache()
                }
            }

            private func effectiveActiveLimit() -> Int {
                let limits = activeScopes.values.map(\.activeCacheLimit)
                return limits.max() ?? effectiveIdleLimit()
            }

            private func idlePolicyConfiguration() -> GPUMemoryConfiguration {
                knownConfigs.max(by: { $0.idleCacheLimit < $1.idleCacheLimit })
                    ?? GPUMemoryConfiguration.automatic
            }

            private func effectiveIdleLimit() -> Int {
                idlePolicyConfiguration().idleCacheLimit
            }

            private func shouldClearOnEviction() -> Bool {
                idlePolicyConfiguration().clearCacheOnEviction
            }
        }

        private static let sessionKVCache = SessionKVStore()
        private static let sessionGenerationGate = SessionGenerationGate()

        /// The model identifier.
        public let modelId: String

        /// The Hub API instance for downloading models.
        public let hub: HubApi?

        /// The local directory containing the model files.
        public let directory: URL?

        /// GPU memory behavior used for this model's generation scopes.
        public let gpuMemory: GPUMemoryConfiguration

        /// Creates an MLX language model.
        ///
        /// - Parameters:
        ///   - modelId: The model identifier (for example, "mlx-community/Llama-3.2-3B-Instruct-4bit").
        ///   - hub: An optional Hub API instance for downloading models. If not provided, the default Hub API is used.
        ///   - directory: An optional local directory URL containing the model files. If provided, the model is loaded from this directory instead of downloading.
        ///   - gpuMemory: The GPU-memory behavior used for this model's active and idle phases.
        public init(
            modelId: String,
            hub: HubApi? = nil,
            directory: URL? = nil,
            gpuMemory: GPUMemoryConfiguration = .automatic
        ) {
            self.modelId = modelId
            self.hub = hub
            self.directory = directory
            self.gpuMemory = gpuMemory
            GPUMemoryManager.shared.register(gpuMemory)
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
            Self.removeSessionCaches(forModelKey: modelSessionCacheKey())
            GPUMemoryManager.shared.evictIfSafe()
        }

        /// Removes all MLX models from the shared cache and cancels in-flight loads.
        public static func removeAllFromCache() async {
            await modelCache.removeAllAndCancel()
            sessionKVCache.removeAll()
            GPUMemoryManager.shared.evictIfSafe()
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

        private static func sessionKey(model: MLXLanguageModel) -> String {
            let directoryKey = model.directory?.absoluteString ?? ""
            return "\(model.modelId)|\(directoryKey)"
        }

        private func modelSessionCacheKey() -> String {
            Self.sessionKey(model: self)
        }

        private func getSessionCache(for session: LanguageModelSession) -> SessionCacheEntry? {
            Self.sessionKVCache.entry(for: session, modelKey: modelSessionCacheKey())
        }

        private func setSessionCache(_ entry: SessionCacheEntry, for session: LanguageModelSession) {
            Self.sessionKVCache.set(entry, for: session, modelKey: modelSessionCacheKey())
        }

        private func removeSessionCache(for session: LanguageModelSession) {
            Self.sessionKVCache.removeEntry(for: session, modelKey: modelSessionCacheKey())
        }

        private static func removeSessionCaches(forModelKey modelKey: String) {
            sessionKVCache.removeEntries(forModelKey: modelKey)
        }

        private static func concurrentSessionError() -> LanguageModelSession.GenerationError {
            .concurrentRequests(
                .init(
                    debugDescription:
                        "Concurrent requests on the same LanguageModelSession are not supported for MLX due to cache and memory management constraints."
                )
            )
        }

        private static func maxToolIterationsExceededError(limit: Int) -> LanguageModelSession.GenerationError {
            .decodingFailure(
                .init(
                    debugDescription:
                        "Exceeded maximum tool iterations (\(limit)) while processing MLX tool calls."
                )
            )
        }

        private static func repeatedToolCallLoopError() -> LanguageModelSession.GenerationError {
            .decodingFailure(
                .init(
                    debugDescription:
                        "Detected repeated MLX tool-call signature and aborted to avoid an infinite tool loop."
                )
            )
        }

        private static func acquireGenerationSlot(for session: LanguageModelSession) -> Bool {
            sessionGenerationGate.acquire(session: session)
        }

        private static func releaseGenerationSlot(for session: LanguageModelSession) {
            sessionGenerationGate.release(session: session)
        }

        private func cacheSignature(from parameters: MLXLMCommon.GenerateParameters) -> CacheConfigSignature {
            CacheConfigSignature(
                maxKVSize: parameters.maxKVSize,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        private func tokens(from input: MLXLMCommon.LMInput) -> [Int32] {
            input.text.tokens.asArray(Int32.self)
        }

        private func isCacheHit(
            entry: SessionCacheEntry,
            currentTokens: [Int32],
            signature: CacheConfigSignature,
            lmInput: MLXLMCommon.LMInput
        ) -> Bool {
            guard lmInput.image == nil, lmInput.video == nil else {
                return false
            }
            guard entry.cacheConfigSignature == signature else {
                return false
            }
            guard entry.prefillTokenCount > 0, currentTokens.count > entry.prefillTokenCount else {
                return false
            }
            guard entry.prefixTokens.count == entry.prefillTokenCount else {
                return false
            }
            return currentTokens.starts(with: entry.prefixTokens)
        }

        private func resolveCache(
            session: LanguageModelSession,
            lmInput: MLXLMCommon.LMInput,
            generateParameters: MLXLMCommon.GenerateParameters,
            context: ModelContext
        ) -> (cache: [MLXLMCommon.KVCache], input: MLXLMCommon.LMInput, fullTokens: [Int32]) {
            let signature = cacheSignature(from: generateParameters)
            let fullTokens = tokens(from: lmInput)
            let existingEntry = getSessionCache(for: session)

            if let existingEntry,
                isCacheHit(entry: existingEntry, currentTokens: fullTokens, signature: signature, lmInput: lmInput)
            {
                let cachedCount = existingEntry.prefillTokenCount
                let newTokens = lmInput.text.tokens[cachedCount...]
                let newMask = lmInput.text.mask?[cachedCount...]
                let partialText = MLXLMCommon.LMInput.Text(tokens: newTokens, mask: newMask)
                return (existingEntry.kvCache, MLXLMCommon.LMInput(text: partialText), fullTokens)
            }

            if existingEntry != nil {
                removeSessionCache(for: session)
            }

            let newCache = context.model.newCache(parameters: generateParameters)
            return (newCache, lmInput, fullTokens)
        }

        private func storeSessionCache(
            cache: [MLXLMCommon.KVCache],
            fullTokens: [Int32],
            generateParameters: MLXLMCommon.GenerateParameters,
            session: LanguageModelSession
        ) {
            let offset = cache.first?.offset ?? 0
            let prefillCount = max(0, min(offset, fullTokens.count))
            guard prefillCount > 0 else {
                removeSessionCache(for: session)
                return
            }

            let prefixTokens = Array(fullTokens.prefix(prefillCount))
            let entry = SessionCacheEntry(
                kvCache: cache,
                prefillTokenCount: prefillCount,
                prefixTokens: prefixTokens,
                cacheConfigSignature: cacheSignature(from: generateParameters)
            )
            setSessionCache(entry, for: session)
        }

        private func beginGenerationScope() -> UUID {
            GPUMemoryManager.shared.markActive(gpuMemory)
        }

        private func endGenerationScope(_ id: UUID) {
            GPUMemoryManager.shared.markIdle(scope: id)
        }

        private func mlxToolSpecs(for session: LanguageModelSession) -> [ToolSpec]? {
            session.tools.isEmpty ? nil : session.tools.map { convertToolToMLXSpec($0) }
        }

        private func makeUserInput(
            session: LanguageModelSession,
            fallbackPrompt: String,
            tools: [ToolSpec]?
        ) -> MLXLMCommon.UserInput {
            let chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: fallbackPrompt)
            return MLXLMCommon.UserInput(
                chat: chat,
                processing: .init(resize: .init(width: 512, height: 512)),
                tools: tools
            )
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            guard Self.acquireGenerationSlot(for: session) else {
                throw Self.concurrentSessionError()
            }
            defer { Self.releaseGenerationSlot(for: session) }

            // Get cached or load fresh ModelContext
            let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)
            let generationScope = beginGenerationScope()
            defer { endGenerationScope(generationScope) }

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

            let toolSpecs = mlxToolSpecs(for: session)

            // Map AnyLanguageModel GenerationOptions to MLX GenerateParameters
            let generateParameters = toGenerateParameters(options)

            // Build chat history from full transcript
            var chat = convertTranscriptToMLXChat(session: session, fallbackPrompt: prompt.description)

            var allTextChunks: [String] = []
            var allEntries: [Transcript.Entry] = []
            let maxToolIterations = 8
            var toolIteration = 0
            var previousToolCallSignature: String?

            // Loop until no more tool calls
            while true {
                // Build user input with current chat history and tools
                let userInput = MLXLMCommon.UserInput(
                    chat: chat,
                    processing: .init(resize: .init(width: 512, height: 512)),
                    tools: toolSpecs
                )
                let lmInput = try await context.processor.prepare(input: userInput)
                let resolved = resolveCache(
                    session: session,
                    lmInput: lmInput,
                    generateParameters: generateParameters,
                    context: context
                )

                // Generate
                let stream = try MLXLMCommon.generate(
                    input: resolved.input,
                    cache: resolved.cache,
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
                storeSessionCache(
                    cache: resolved.cache,
                    fullTokens: resolved.fullTokens,
                    generateParameters: generateParameters,
                    session: session
                )

                let assistantText = chunks.joined()
                allTextChunks.append(assistantText)

                // Add assistant response to chat history
                if !assistantText.isEmpty {
                    chat.append(.assistant(assistantText))
                }

                // If there are tool calls, execute them and continue
                if !collectedToolCalls.isEmpty {
                    toolIteration += 1
                    if toolIteration > maxToolIterations {
                        let unresolvedCalls = try makeTranscriptToolCalls(from: collectedToolCalls)
                        allEntries.append(Transcript.Entry.toolCalls(Transcript.ToolCalls(unresolvedCalls)))
                        throw Self.maxToolIterationsExceededError(limit: maxToolIterations)
                    }

                    let signature =
                        collectedToolCalls
                        .map { "\($0.function.name):\($0.function.arguments)" }
                        .joined(separator: "|")
                    if signature == previousToolCallSignature {
                        let unresolvedCalls = try makeTranscriptToolCalls(from: collectedToolCalls)
                        allEntries.append(Transcript.Entry.toolCalls(Transcript.ToolCalls(unresolvedCalls)))
                        throw Self.repeatedToolCallLoopError()
                    }
                    previousToolCallSignature = signature

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
            guard Self.acquireGenerationSlot(for: session) else {
                let error = Self.concurrentSessionError()
                let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                    .init { continuation in
                        continuation.finish(throwing: error)
                    }
                return LanguageModelSession.ResponseStream(stream: stream)
            }

            let modelId = self.modelId
            let hub = self.hub
            let directory = self.directory
            let gpuMemory = self.gpuMemory

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> = .init {
                continuation in
                let didEndScope = Locked(false)
                let didReleaseGenerationSlot = Locked(false)
                let generationScope = GPUMemoryManager.shared.markActive(gpuMemory)

                let task = Task { @Sendable in
                    func finishScope() {
                        didEndScope.withLock { done in
                            if !done {
                                GPUMemoryManager.shared.markIdle(scope: generationScope)
                                done = true
                            }
                        }
                    }

                    func finishGenerationSlot() {
                        didReleaseGenerationSlot.withLock { done in
                            if !done {
                                Self.releaseGenerationSlot(for: session)
                                done = true
                            }
                        }
                    }

                    do {
                        // Get cached or load fresh ModelContext
                        let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)

                        // Build chat inside task to avoid Sendable issues
                        let generateParameters = toGenerateParameters(options)
                        let userInput = makeUserInput(
                            session: session,
                            fallbackPrompt: prompt.description,
                            tools: nil
                        )
                        let lmInput = try await context.processor.prepare(input: userInput)
                        let resolved = resolveCache(
                            session: session,
                            lmInput: lmInput,
                            generateParameters: generateParameters,
                            context: context
                        )

                        let mlxStream = try MLXLMCommon.generate(
                            input: resolved.input,
                            cache: resolved.cache,
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

                        storeSessionCache(
                            cache: resolved.cache,
                            fullTokens: resolved.fullTokens,
                            generateParameters: generateParameters,
                            session: session
                        )
                        finishScope()
                        finishGenerationSlot()
                        continuation.finish()
                    } catch {
                        finishScope()
                        finishGenerationSlot()
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in
                    didEndScope.withLock { done in
                        if !done {
                            GPUMemoryManager.shared.markIdle(scope: generationScope)
                            done = true
                        }
                    }
                    didReleaseGenerationSlot.withLock { done in
                        if !done {
                            Self.releaseGenerationSlot(for: session)
                            done = true
                        }
                    }
                    task.cancel()
                }
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
                guard Self.acquireGenerationSlot(for: session) else {
                    return
                }
                defer { Self.releaseGenerationSlot(for: session) }

                let generationScope = beginGenerationScope()
                defer { endGenerationScope(generationScope) }

                do {
                    let context = try await loadContext(modelId: modelId, hub: hub, directory: directory)
                    guard let instructions = session.instructions?.description, !instructions.isEmpty else {
                        return
                    }

                    let toolSpecs = mlxToolSpecs(for: session)

                    let params = toGenerateParameters(.init())
                    let newCache = context.model.newCache(parameters: params)
                    let userInput = MLXLMCommon.UserInput(
                        chat: [.init(role: .system, content: instructions)],
                        processing: .init(resize: .init(width: 512, height: 512)),
                        tools: toolSpecs
                    )
                    let lmInput = try await context.processor.prepare(input: userInput)
                    _ = try context.model.prepare(lmInput, cache: newCache, windowSize: params.prefillStepSize)
                    storeSessionCache(
                        cache: newCache,
                        fullTokens: tokens(from: lmInput),
                        generateParameters: params,
                        session: session
                    )
                } catch {
                    // Ignore errors during prewarm
                }
            }
        }
    }

    // MARK: - Options Mapping

    private func toGenerateParameters(_ options: GenerationOptions) -> MLXLMCommon.GenerateParameters {
        let custom = options[custom: MLXLanguageModel.self]
        return MLXLMCommon.GenerateParameters(
            maxTokens: options.maximumResponseTokens,
            maxKVSize: custom?.maxKVSize,
            kvBits: custom?.kvBits,
            kvGroupSize: custom?.kvGroupSize ?? 64,
            quantizedKVStart: custom?.quantizedKVStart ?? 0,
            temperature: Float(options.temperature ?? 0.6),
            topP: 1.0,
            repetitionPenalty: nil,
            repetitionContextSize: 20
        )
    }

    /// Builds MLX parameters tuned for structured generation.
    private func toStructuredGenerateParameters(_ options: GenerationOptions) -> MLXLMCommon.GenerateParameters {
        let custom = options[custom: MLXLanguageModel.self]
        return MLXLMCommon.GenerateParameters(
            maxTokens: options.maximumResponseTokens,
            maxKVSize: custom?.maxKVSize,
            kvBits: custom?.kvBits,
            kvGroupSize: custom?.kvGroupSize ?? 64,
            quantizedKVStart: custom?.quantizedKVStart ?? 0,
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

    private func makeTranscriptToolCalls(
        from toolCalls: [MLXLMCommon.ToolCall]
    ) throws -> [Transcript.ToolCall] {
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
        return transcriptCalls
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

        let transcriptCalls = try makeTranscriptToolCalls(from: toolCalls)

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
