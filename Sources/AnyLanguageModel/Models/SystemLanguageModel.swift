#if canImport(FoundationModels)
    import FoundationModels
    import Foundation
    import PartialJSONDecoder

    import JSONSchema

    /// A language model that uses Apple Intelligence.
    ///
    /// Use this model to generate text using on-device language models provided by Apple.
    /// This model runs entirely on-device and doesn't send data to external servers.
    ///
    /// ```swift
    /// let model = SystemLanguageModel()
    /// ```
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    public actor SystemLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        public typealias UnavailableReason = FoundationModels.SystemLanguageModel.Availability.UnavailableReason

        let systemModel: FoundationModels.SystemLanguageModel

        /// The default system language model.
        public static var `default`: SystemLanguageModel {
            SystemLanguageModel()
        }

        /// Creates the default system language model.
        public init() {
            self.systemModel = FoundationModels.SystemLanguageModel.default
        }

        /// Creates a system language model for a specific use case.
        ///
        /// - Parameters:
        ///   - useCase: The intended use case for generation.
        ///   - guardrails: Safety guardrails to apply during generation.
        public init(
            useCase: FoundationModels.SystemLanguageModel.UseCase = .general,
            guardrails: FoundationModels.SystemLanguageModel.Guardrails = FoundationModels.SystemLanguageModel
                .Guardrails.default
        ) {
            self.systemModel = FoundationModels.SystemLanguageModel(useCase: useCase, guardrails: guardrails)
        }

        /// Creates a system language model with a custom adapter.
        ///
        /// - Parameters:
        ///   - adapter: The adapter to use with the base model.
        ///   - guardrails: Safety guardrails to apply during generation.
        public init(
            adapter: FoundationModels.SystemLanguageModel.Adapter,
            guardrails: FoundationModels.SystemLanguageModel.Guardrails = .default
        ) {
            self.systemModel = FoundationModels.SystemLanguageModel(adapter: adapter, guardrails: guardrails)
        }

        /// The availability status for the system language model.
        nonisolated public var availability: Availability<UnavailableReason> {
            switch systemModel.availability {
            case .available:
                .available
            case .unavailable(let reason):
                .unavailable(reason)
            }
        }

        nonisolated public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let fmPrompt = prompt.toFoundationModels()
            let fmOptions = options.toFoundationModels()

            let fmSession = FoundationModels.LanguageModelSession(
                model: systemModel,
                tools: session.tools.toFoundationModels(),
                transcript: session.transcript.toFoundationModels(
                    instructions: session.instructions,
                    toolDefinitions: session.tools
                        .filter(\.includesSchemaInInstructions)
                        .map { Transcript.ToolDefinition(tool: $0) }
                )
            )

            if type == String.self {
                let fmResponse = try await fmSession.respond(to: fmPrompt, options: fmOptions)
                let generatedContent = GeneratedContent(fmResponse.content)
                return LanguageModelSession.Response(
                    content: fmResponse.content as! Content,
                    rawContent: generatedContent,
                    transcriptEntries: []
                )
            } else {
                // For non-String types, use schema-based generation
                let schema = FoundationModels.GenerationSchema(type.generationSchema)
                let fmResponse = try await fmSession.respond(
                    to: fmPrompt,
                    schema: schema,
                    includeSchemaInPrompt: includeSchemaInPrompt,
                    options: fmOptions
                )

                func finalize(content: Content) -> LanguageModelSession.Response<Content> {
                    let normalizedRaw = content.generatedContent
                    if let jsonValue = try? JSONValue(normalizedRaw),
                        case .array(let values) = jsonValue,
                        values.isEmpty,
                        let placeholder = placeholderContent(for: type)
                    {
                        return LanguageModelSession.Response(
                            content: placeholder.content,
                            rawContent: placeholder.rawContent,
                            transcriptEntries: []
                        )
                    }
                    return LanguageModelSession.Response(
                        content: content,
                        rawContent: normalizedRaw,
                        transcriptEntries: []
                    )
                }

                do {
                    let generatedContent = try GeneratedContent(fmResponse.content)
                    let content = try type.init(generatedContent)

                    return finalize(content: content)
                } catch {
                    // Attempt partial JSON decoding before surfacing an error.
                    let decoder = PartialJSONDecoder()
                    let jsonString = fmResponse.content.jsonString
                    if let partialContent = try? decoder.decode(GeneratedContent.self, from: jsonString).value,
                        let content = try? type.init(partialContent)
                    {
                        return finalize(content: content)
                    }
                    if let placeholder = placeholderContent(for: type) {
                        return finalize(content: placeholder.content)
                    }
                    throw error
                }
            }
        }

        nonisolated public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            let fmPrompt = prompt.toFoundationModels()
            let fmOptions = options.toFoundationModels()

            let fmSession = FoundationModels.LanguageModelSession(
                model: systemModel,
                tools: session.tools.toFoundationModels(),
                transcript: session.transcript.toFoundationModels(
                    instructions: session.instructions,
                    toolDefinitions: session.tools
                        .filter(\.includesSchemaInInstructions)
                        .map { Transcript.ToolDefinition(tool: $0) }
                )
            )

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, Error> =
                AsyncThrowingStream { continuation in

                    func accumulateText(
                        _ chunkText: String,
                        accumulatedText: inout String,
                        lastLength: inout Int
                    ) {
                        if chunkText.count >= lastLength, chunkText.hasPrefix(accumulatedText) {
                            let startIdx = chunkText.index(chunkText.startIndex, offsetBy: lastLength)
                            let delta = String(chunkText[startIdx...])
                            accumulatedText += delta
                            lastLength = chunkText.count
                        } else if chunkText.hasPrefix(accumulatedText) {
                            accumulatedText = chunkText
                            lastLength = chunkText.count
                        } else if accumulatedText.hasPrefix(chunkText) {
                            accumulatedText = chunkText
                            lastLength = chunkText.count
                        } else {
                            accumulatedText += chunkText
                            lastLength = accumulatedText.count
                        }
                    }

                    func processStringStream() async {
                        let fmStream: FoundationModels.LanguageModelSession.ResponseStream<String> =
                            fmSession.streamResponse(to: fmPrompt, options: fmOptions)

                        var accumulatedText = ""
                        do {
                            var lastLength = 0
                            for try await snapshot in fmStream {
                                var chunkText: String = snapshot.content

                                // Handle "null" from FoundationModels as first temp result
                                if chunkText == "null" && accumulatedText == "" {
                                    chunkText = ""
                                }

                                accumulateText(
                                    chunkText,
                                    accumulatedText: &accumulatedText,
                                    lastLength: &lastLength
                                )

                                let raw = GeneratedContent(accumulatedText)
                                let snapshotContent = (accumulatedText as! Content).asPartiallyGenerated()
                                continuation.yield(.init(content: snapshotContent, rawContent: raw))
                            }
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }

                    func processStructuredStream() async {
                        let schema = FoundationModels.GenerationSchema(type.generationSchema)
                        let partialDecoder = PartialJSONDecoder()
                        let fmStream = fmSession.streamResponse(
                            to: fmPrompt,
                            schema: schema,
                            includeSchemaInPrompt: includeSchemaInPrompt,
                            options: fmOptions
                        )

                        func processTextFallback() async {
                            let fmTextStream: FoundationModels.LanguageModelSession.ResponseStream<String> =
                                fmSession.streamResponse(to: fmPrompt, options: fmOptions)

                            var accumulatedText = ""
                            var didYield = false
                            do {
                                var lastLength = 0
                                for try await snapshot in fmTextStream {
                                    var chunkText: String = snapshot.content
                                    if chunkText == "null" && accumulatedText.isEmpty {
                                        chunkText = ""
                                    }

                                    accumulateText(
                                        chunkText,
                                        accumulatedText: &accumulatedText,
                                        lastLength: &lastLength
                                    )

                                    let jsonString = accumulatedText
                                    if let partialContent = try? partialDecoder.decode(
                                        GeneratedContent.self,
                                        from: jsonString
                                    )
                                    .value {
                                        let partial: Content.PartiallyGenerated? = try? .init(partialContent)
                                        if let partial {
                                            continuation.yield(.init(content: partial, rawContent: partialContent))
                                            didYield = true
                                        }
                                    }
                                }
                                if !didYield, let placeholder = placeholderPartialContent(for: type) {
                                    continuation.yield(
                                        .init(content: placeholder.content, rawContent: placeholder.rawContent)
                                    )
                                }
                                continuation.finish()
                            } catch {
                                if !didYield, let placeholder = placeholderPartialContent(for: type) {
                                    continuation.yield(
                                        .init(content: placeholder.content, rawContent: placeholder.rawContent)
                                    )
                                }
                                continuation.finish(throwing: error)
                            }
                        }

                        var didYield = false
                        do {
                            for try await snapshot in fmStream {
                                let jsonString = snapshot.content.jsonString
                                let raw =
                                    (try? GeneratedContent(snapshot.content))
                                    ?? (try? GeneratedContent(json: jsonString))
                                    ?? GeneratedContent(jsonString)

                                // Prefer partial decoding so we can surface intermediate snapshots.
                                if let partialContent = try? partialDecoder.decode(
                                    GeneratedContent.self,
                                    from: jsonString
                                )
                                .value {
                                    let partial: Content.PartiallyGenerated? = try? .init(partialContent)
                                    if let partial {
                                        continuation.yield(.init(content: partial, rawContent: partialContent))
                                        didYield = true
                                        continue
                                    }
                                }

                                // Fallback to full conversion when partial decoding isn't possible.
                                if let value = try? type.init(raw) {
                                    let snapshotContent = value.asPartiallyGenerated()
                                    continuation.yield(.init(content: snapshotContent, rawContent: raw))
                                    didYield = true
                                }
                            }
                            if !didYield, let placeholder = placeholderPartialContent(for: type) {
                                continuation.yield(
                                    .init(content: placeholder.content, rawContent: placeholder.rawContent)
                                )
                            }
                            continuation.finish()
                        } catch {
                            if didYield {
                                continuation.finish(throwing: error)
                            } else {
                                await processTextFallback()
                            }
                        }
                    }

                    let streamingTask: _Concurrency.Task<Void, Never> = _Concurrency.Task(priority: nil) {
                        if type == String.self {
                            await processStringStream()
                        } else {
                            await processStructuredStream()
                        }
                    }
                    continuation.onTermination = { _ in streamingTask.cancel() }
                }

            return LanguageModelSession.ResponseStream(stream: stream)
        }

        nonisolated public func logFeedbackAttachment(
            within session: LanguageModelSession,
            sentiment: LanguageModelFeedback.Sentiment?,
            issues: [LanguageModelFeedback.Issue],
            desiredOutput: Transcript.Entry?
        ) -> Data {
            let fmSession = FoundationModels.LanguageModelSession(
                model: systemModel,
                tools: session.tools.toFoundationModels(),
                instructions: session.instructions?.toFoundationModels()
            )

            let fmSentiment = sentiment?.toFoundationModels()
            let fmIssues = issues.map { $0.toFoundationModels() }
            let fmDesiredOutput: FoundationModels.Transcript.Entry? = nil

            return fmSession.logFeedbackAttachment(
                sentiment: fmSentiment,
                issues: fmIssues,
                desiredOutput: fmDesiredOutput
            )
        }

    }

    // MARK: - Helpers

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Prompt {
        fileprivate func toFoundationModels() -> FoundationModels.Prompt {
            FoundationModels.Prompt(self.description)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Instructions {
        fileprivate func toFoundationModels() -> FoundationModels.Instructions {
            FoundationModels.Instructions(self.description)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension GenerationOptions {
        fileprivate func toFoundationModels() -> FoundationModels.GenerationOptions {
            var options = FoundationModels.GenerationOptions()

            if let temperature = self.temperature {
                options.temperature = temperature
            }

            // Note: FoundationModels.GenerationOptions may not have all properties
            // Only set those that are available

            return options
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension LanguageModelFeedback.Sentiment {
        fileprivate func toFoundationModels() -> FoundationModels.LanguageModelFeedback.Sentiment {
            switch self {
            case .positive: .positive
            case .negative: .negative
            case .neutral: .neutral
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension LanguageModelFeedback.Issue {
        fileprivate func toFoundationModels() -> FoundationModels.LanguageModelFeedback.Issue {
            FoundationModels.LanguageModelFeedback.Issue(
                category: self.category.toFoundationModels(),
                explanation: self.explanation
            )
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension LanguageModelFeedback.Issue.Category {
        fileprivate func toFoundationModels() -> FoundationModels.LanguageModelFeedback.Issue.Category {
            switch self {
            case .unhelpful: .unhelpful
            case .tooVerbose: .tooVerbose
            case .didNotFollowInstructions: .didNotFollowInstructions
            case .incorrect: .incorrect
            case .stereotypeOrBias: .stereotypeOrBias
            case .suggestiveOrSexual: .suggestiveOrSexual
            case .vulgarOrOffensive: .vulgarOrOffensive
            case .triggeredGuardrailUnexpectedly: .triggeredGuardrailUnexpectedly
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == (any Tool) {
        fileprivate func toFoundationModels() -> [any FoundationModels.Tool] {
            map { AnyToolWrapper($0) }
        }
    }

    /// A type-erased wrapper that bridges any `Tool` to `FoundationModels.Tool`.
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private struct AnyToolWrapper: FoundationModels.Tool {
        typealias Arguments = FoundationModels.GeneratedContent
        typealias Output = String

        let name: String
        let description: String
        let parameters: FoundationModels.GenerationSchema
        let includesSchemaInInstructions: Bool

        private let wrappedTool: any Tool

        init(_ tool: any Tool) {
            self.wrappedTool = tool
            self.name = tool.name
            self.description = tool.description
            self.parameters = FoundationModels.GenerationSchema(tool.parameters)
            self.includesSchemaInInstructions = tool.includesSchemaInInstructions
        }

        func call(arguments: FoundationModels.GeneratedContent) async throws -> Output {
            let output = try await wrappedTool.callFunction(arguments: arguments)
            return output.promptRepresentation.description
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension FoundationModels.GenerationSchema {
        internal init(_ content: AnyLanguageModel.GenerationSchema) {
            let resolvedSchema = content.withResolvedRoot() ?? content

            // Convert the GenerationSchema into a DynamicGenerationSchema, preserving $defs
            let rawParameters = try? JSONValue(resolvedSchema)

            if case .object(var rootObject) = rawParameters {
                // Extract dependencies from $defs and remove from the root payload
                let defs = rootObject.removeValue(forKey: "$defs")?.objectValue ?? [:]

                // Convert root schema
                if let rootData = try? JSONEncoder().encode(JSONValue.object(rootObject)),
                    let rootJSONSchema = try? JSONDecoder().decode(JSONSchema.self, from: rootData)
                {
                    let rootDynamicSchema = convertToDynamicSchema(rootJSONSchema)

                    // Convert each dependency schema
                    let dependencies: [FoundationModels.DynamicGenerationSchema] = defs.compactMap { name, value in
                        guard
                            let defData = try? JSONEncoder().encode(value),
                            let defJSONSchema = try? JSONDecoder().decode(JSONSchema.self, from: defData)
                        else {
                            return nil
                        }
                        return convertToDynamicSchema(defJSONSchema, name: name)
                    }

                    if let schema = try? FoundationModels.GenerationSchema(
                        root: rootDynamicSchema,
                        dependencies: dependencies
                    ) {
                        self = schema
                        return
                    }
                }
            }

            // Fallback to a minimal string schema if conversion fails
            self = FoundationModels.GenerationSchema(
                type: String.self,
                properties: []
            )
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension FoundationModels.GeneratedContent {
        internal init(_ content: AnyLanguageModel.GeneratedContent) throws {
            try self.init(json: content.jsonString)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension AnyLanguageModel.GeneratedContent {
        internal init(_ content: FoundationModels.GeneratedContent) throws {
            try self.init(json: content.jsonString)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Tool {
        fileprivate func callFunction(arguments: FoundationModels.GeneratedContent) async throws
            -> any PromptRepresentable
        {
            let content = try GeneratedContent(arguments)
            return try await call(arguments: Self.Arguments(content))
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func convertToDynamicSchema(
        _ jsonSchema: JSONSchema,
        name: String? = nil
    ) -> FoundationModels.DynamicGenerationSchema {
        switch jsonSchema {
        case .object(_, _, _, _, _, _, properties: let properties, required: let required, _):
            let schemaProperties = properties.compactMap { key, value in
                convertToProperty(key: key, schema: value, required: required)
            }
            return .init(name: name ?? "", description: jsonSchema.description, properties: schemaProperties)

        case .string(_, _, _, _, _, _, _, _, pattern: let pattern, _):
            var guides: [FoundationModels.GenerationGuide<String>] = []
            if let values = jsonSchema.enum?.compactMap(\.stringValue), !values.isEmpty {
                guides.append(.anyOf(values))
            }
            if let value = jsonSchema.const?.stringValue {
                guides.append(.constant(value))
            }
            if let pattern, let regex = try? Regex(pattern) {
                guides.append(.pattern(regex))
            }
            return .init(type: String.self, guides: guides)

        case .integer(_, _, _, _, _, _, minimum: let minimum, maximum: let maximum, _, _, _):
            if let enumValues = jsonSchema.enum {
                let enumsSchema = enumValues.compactMap { convertConstToSchema($0) }
                return .init(name: name ?? "", anyOf: enumsSchema)
            }

            var guides: [FoundationModels.GenerationGuide<Int>] = []
            if let min = minimum {
                guides.append(.minimum(min))
            }
            if let max = maximum {
                guides.append(.maximum(max))
            }
            if let value = jsonSchema.const?.intValue {
                guides.append(.range(value ... value))
            }
            return .init(type: Int.self, guides: guides)

        case .number(_, _, _, _, _, _, minimum: let minimum, maximum: let maximum, _, _, _):
            if let enumValues = jsonSchema.enum {
                let enumsSchema = enumValues.compactMap { convertConstToSchema($0) }
                return .init(name: name ?? "", anyOf: enumsSchema)
            }

            var guides: [FoundationModels.GenerationGuide<Double>] = []
            if let min = minimum {
                guides.append(.minimum(min))
            }
            if let max = maximum {
                guides.append(.maximum(max))
            }
            if let value = jsonSchema.const?.doubleValue {
                guides.append(.range(value ... value))
            }
            return .init(type: Double.self, guides: guides)

        case .boolean:
            return .init(type: Bool.self)

        case .anyOf(let schemas):
            return .init(name: name ?? "", anyOf: schemas.map { convertToDynamicSchema($0) })

        case .array(_, _, _, _, _, _, items: let items, minItems: let minItems, maxItems: let maxItems, _):
            let itemsSchema =
                items.map { convertToDynamicSchema($0) }
                ?? FoundationModels.DynamicGenerationSchema(type: String.self)
            return .init(arrayOf: itemsSchema, minimumElements: minItems, maximumElements: maxItems)

        case .reference(let name):
            return .init(referenceTo: name)

        case .allOf, .oneOf, .not, .null, .empty, .any:
            return .init(type: String.self)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func convertToProperty(
        key: String,
        schema: JSONSchema,
        required: [String]
    ) -> FoundationModels.DynamicGenerationSchema.Property {
        .init(
            name: key,
            description: schema.description,
            schema: convertToDynamicSchema(schema),
            isOptional: !required.contains(key)
        )
    }

    /// Converts a JSON constant value to a DynamicGenerationSchema.
    /// Only handles scalar types (int, double, string); returns nil for null, object, bool, and array.
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func convertConstToSchema(_ value: JSONValue) -> FoundationModels.DynamicGenerationSchema? {
        switch value {
        case .int(let intValue):
            .init(type: Int.self, guides: [.range(intValue ... intValue)])
        case .double(let doubleValue):
            .init(type: Double.self, guides: [.range(doubleValue ... doubleValue)])
        case .string(let stringValue):
            .init(type: String.self, guides: [.constant(stringValue)])
        case .null, .object, .bool, .array:
            nil
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Transcript {
        fileprivate func toFoundationModels(
            instructions: AnyLanguageModel.Instructions?,
            toolDefinitions: [Transcript.ToolDefinition]
        ) -> FoundationModels.Transcript {
            var fmEntries: [FoundationModels.Transcript.Entry] = []

            // Add instructions entry if provided and not already in transcript
            if let instructions = instructions {
                let hasInstructions =
                    self.first.map { entry in
                        if case .instructions = entry { return true } else { return false }
                    } ?? false

                if !hasInstructions {
                    let fmInstructions = FoundationModels.Transcript.Instructions(
                        segments: [.text(.init(content: instructions.description))],
                        toolDefinitions: toolDefinitions.toFoundationModels()
                    )
                    fmEntries.append(.instructions(fmInstructions))
                }
            }

            // Convert each entry
            for entry in self {
                switch entry {
                case .instructions(let instr):
                    let fmSegments = instr.segments.toFoundationModels()
                    let fmToolDefinitions = instr.toolDefinitions.toFoundationModels()
                    let fmInstructions = FoundationModels.Transcript.Instructions(
                        segments: fmSegments,
                        toolDefinitions: fmToolDefinitions
                    )
                    fmEntries.append(.instructions(fmInstructions))

                case .prompt(let prompt):
                    let fmSegments = prompt.segments.toFoundationModels()
                    let fmPrompt = FoundationModels.Transcript.Prompt(
                        segments: fmSegments
                    )
                    fmEntries.append(.prompt(fmPrompt))

                case .response(let response):
                    let fmSegments = response.segments.toFoundationModels()
                    let fmResponse = FoundationModels.Transcript.Response(
                        assetIDs: response.assetIDs,
                        segments: fmSegments
                    )
                    fmEntries.append(.response(fmResponse))

                case .toolCalls(let toolCalls):
                    let fmCalls = toolCalls.compactMap { call -> FoundationModels.Transcript.ToolCall? in
                        guard let fmArguments = try? FoundationModels.GeneratedContent(call.arguments) else {
                            return nil
                        }
                        return FoundationModels.Transcript.ToolCall(
                            id: call.id,
                            toolName: call.toolName,
                            arguments: fmArguments
                        )
                    }
                    let fmToolCalls = FoundationModels.Transcript.ToolCalls(id: toolCalls.id, fmCalls)
                    fmEntries.append(.toolCalls(fmToolCalls))

                case .toolOutput(let toolOutput):
                    let fmSegments = toolOutput.segments.toFoundationModels()
                    let fmToolOutput = FoundationModels.Transcript.ToolOutput(
                        id: toolOutput.id,
                        toolName: toolOutput.toolName,
                        segments: fmSegments
                    )
                    fmEntries.append(.toolOutput(fmToolOutput))
                }
            }

            return FoundationModels.Transcript(entries: fmEntries)
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == Transcript.Segment {
        fileprivate func toFoundationModels() -> [FoundationModels.Transcript.Segment] {
            compactMap { segment -> FoundationModels.Transcript.Segment? in
                switch segment {
                case .text(let textSegment):
                    return .text(.init(id: textSegment.id, content: textSegment.content))
                case .structure(let structuredSegment):
                    guard let fmContent = try? FoundationModels.GeneratedContent(structuredSegment.content) else {
                        return nil
                    }
                    return .structure(
                        .init(
                            id: structuredSegment.id,
                            source: structuredSegment.source,
                            content: fmContent
                        )
                    )
                case .image:
                    // FoundationModels Transcript does not support image segments
                    return nil
                }
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    extension Array where Element == Transcript.ToolDefinition {
        fileprivate func toFoundationModels() -> [FoundationModels.Transcript.ToolDefinition] {
            map { toolDef in
                FoundationModels.Transcript.ToolDefinition(
                    name: toolDef.name,
                    description: toolDef.description,
                    parameters: FoundationModels.GenerationSchema(toolDef.parameters)
                )
            }
        }
    }

    // MARK: - Placeholder Helpers

    /// Generates minimal partial content when structured output is missing or invalid.
    private func placeholderPartialContent<Content: Generable>(
        for type: Content.Type
    ) -> (content: Content.PartiallyGenerated, rawContent: GeneratedContent)? {
        let schema = type.generationSchema
        let resolved = schema.withResolvedRoot() ?? schema
        let raw = placeholderGeneratedContent(from: resolved.root, defs: resolved.defs)

        if let partial: Content.PartiallyGenerated = try? .init(raw) {
            return (partial, raw)
        }
        if let value = try? Content(raw) {
            return (value.asPartiallyGenerated(), raw)
        }
        return nil
    }

    /// Generates minimal full content when structured output is missing or invalid.
    private func placeholderContent<Content: Generable>(
        for type: Content.Type
    ) -> (content: Content, rawContent: GeneratedContent)? {
        let schema = type.generationSchema
        let resolved = schema.withResolvedRoot() ?? schema
        let raw = placeholderGeneratedContent(from: resolved.root, defs: resolved.defs)

        if let value = try? Content(raw) {
            return (value, raw)
        }
        return nil
    }

    /// Builds a minimal generated content tree from a schema node.
    private func placeholderGeneratedContent(
        from node: GenerationSchema.Node,
        defs: [String: GenerationSchema.Node]
    ) -> GeneratedContent {
        switch node {
        case .object(let obj):
            var properties: Array<(String, GeneratedContent)> = []
            for (key, value) in obj.properties {
                let generated = placeholderGeneratedContent(from: value, defs: defs)
                properties.append((key, generated))
            }
            let convertible: [(String, any ConvertibleToGeneratedContent)] = properties.map {
                ($0.0, $0.1 as any ConvertibleToGeneratedContent)
            }
            return GeneratedContent(
                properties: convertible,
                id: nil,
                uniquingKeysWith: { first, _ in first }
            )

        case .array(let arr):
            let item = placeholderGeneratedContent(from: arr.items, defs: defs)
            let count = max(arr.minItems ?? 1, 1)
            let elements = Array(repeating: item, count: count)
            return GeneratedContent(elements: elements)

        case .string(let str):
            if let first = str.enumChoices?.first {
                return GeneratedContent(first)
            }
            return GeneratedContent("placeholder")

        case .number(let num):
            if num.integerOnly {
                return GeneratedContent(Int(num.minimum ?? 0))
            } else {
                return GeneratedContent(num.minimum ?? 0)
            }

        case .boolean:
            return GeneratedContent(true)

        case .anyOf(let nodes):
            if let first = nodes.first {
                return placeholderGeneratedContent(from: first, defs: defs)
            }
            return GeneratedContent("placeholder")

        case .ref(let name):
            if let node = defs[name] {
                return placeholderGeneratedContent(from: node, defs: defs)
            }
            return GeneratedContent("placeholder")
        }
    }
#endif
