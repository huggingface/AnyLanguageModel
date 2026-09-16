#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS)
    import Foundation
    import FoundationModels

    /// A language model backed by any type that conforms to
    /// `FoundationModels.LanguageModel`.
    ///
    /// On OS 27, Apple's Foundation Models framework opens its `LanguageModel`
    /// protocol to third-party models, and `LanguageModelSession` accepts any
    /// conformer. This wrapper hands such a model to a Foundation Models session
    /// and bridges the result back into AnyLanguageModel, so a model built on
    /// Apple's protocol can be used alongside every other provider here.
    ///
    /// Models that are expensive to construct can be supplied through an async
    /// factory. The factory runs once, on the first request or on an explicit
    /// call to ``load()``, and the caller owns the lifetime through ``unload()``.
    ///
    /// ```swift
    /// let model = FoundationLanguageModel {
    ///     try await MyModel(resourcesAt: url)
    /// }
    /// let session = LanguageModelSession(model: model)
    /// let response = try await session.respond(to: "Hello")
    /// await model.unload()
    /// ```
    @available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *)
    public actor FoundationLanguageModel<Model: FoundationModels.LanguageModel>: LanguageModel {
        public typealias UnavailableReason = Never

        private let makeModel: @Sendable () async throws -> Model
        private var model: Model?

        /// The factory run in progress, if any.
        /// The actor suspends while the factory runs,
        /// so concurrent first requests share this task
        /// instead of each constructing a model.
        private var loadTask: Task<Model, Error>?

        /// Creates a language model around an already constructed model.
        ///
        /// - Parameter model: The Foundation Models conformer to use for generation.
        public init(_ model: Model) {
            self.makeModel = { model }
            self.model = model
        }

        /// Creates a language model whose underlying model is constructed on demand.
        ///
        /// - Parameter makeModel: A factory that constructs the model. It runs once,
        ///   on the first request or on ``load()``, and again after ``unload()``.
        public init(loading makeModel: @escaping @Sendable () async throws -> Model) {
            self.makeModel = makeModel
        }

        /// Whether the underlying model is currently constructed.
        public var isLoaded: Bool {
            model != nil
        }

        /// The capabilities of the underlying model, once it is loaded.
        public var capabilities: FoundationModels.LanguageModelCapabilities? {
            model?.capabilities
        }

        /// Constructs the underlying model if it is not loaded yet.
        public func load() async throws {
            _ = try await loadedModel()
        }

        /// Releases the underlying model. The next request constructs it again.
        ///
        /// A factory still running when this is called is cancelled,
        /// and its result is discarded.
        public func unload() {
            loadTask?.cancel()
            loadTask = nil
            model = nil
        }

        private func loadedModel() async throws -> Model {
            if let model {
                return model
            }
            let task: Task<Model, Error>
            if let loadTask {
                task = loadTask
            } else {
                task = Task { try await makeModel() }
                loadTask = task
            }
            do {
                let model = try await task.value
                // Publish only if unload() did not run while the factory was in flight.
                if loadTask == task {
                    self.model = model
                    loadTask = nil
                }
                return model
            } catch {
                if loadTask == task {
                    loadTask = nil
                }
                throw error
            }
        }

        private func makeSession(
            tools: [any FoundationModels.Tool],
            transcript: FoundationModels.Transcript
        ) async throws -> FoundationModels.LanguageModelSession {
            FoundationModels.LanguageModelSession(
                model: try await loadedModel(),
                tools: tools,
                transcript: transcript
            )
        }

        nonisolated public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            try await respond(
                within: session,
                to: prompt,
                generating: type,
                schema: type.generationSchema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        nonisolated public func respond(
            within session: LanguageModelSession,
            to prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<GeneratedContent> {
            try await respond(
                within: session,
                to: prompt,
                generating: GeneratedContent.self,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        nonisolated private func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let fmTools = session.tools.toFoundationModels()
            let fmTranscript = fmTranscriptDroppingDuplicatePrompt(session.transcript, prompt: prompt)
                .toFoundationModels(
                    instructions: session.instructions,
                    toolDefinitions: session.tools
                        .filter(\.includesSchemaInInstructions)
                        .map { Transcript.ToolDefinition(tool: $0) }
                )
            return try await fmRespond(
                makeSession: { try await self.makeSession(tools: fmTools, transcript: fmTranscript) },
                fmPrompt: prompt.toFoundationModels(),
                fmOptions: options.toFoundationModels(),
                type: type,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt
            )
        }

        nonisolated public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            streamResponse(
                within: session,
                to: prompt,
                generating: type,
                schema: type.generationSchema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        nonisolated public func streamResponse(
            within session: LanguageModelSession,
            to prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<GeneratedContent> {
            streamResponse(
                within: session,
                to: prompt,
                generating: GeneratedContent.self,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        nonisolated private func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            let fmTools = session.tools.toFoundationModels()
            let fmTranscript = fmTranscriptDroppingDuplicatePrompt(session.transcript, prompt: prompt)
                .toFoundationModels(
                    instructions: session.instructions,
                    toolDefinitions: session.tools
                        .filter(\.includesSchemaInInstructions)
                        .map { Transcript.ToolDefinition(tool: $0) }
                )
            return fmStreamResponse(
                makeSession: { try await self.makeSession(tools: fmTools, transcript: fmTranscript) },
                fmPrompt: prompt.toFoundationModels(),
                fmOptions: options.toFoundationModels(),
                type: type,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt
            )
        }
    }
#endif
