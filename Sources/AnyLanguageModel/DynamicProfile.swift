/// Options that configure a model's request context.
///
/// - Note: This API is exclusive to AnyLanguageModel on OS 26 and mirrors
///   Foundation Models 27.
public struct ContextOptions: Sendable, Equatable {
    public enum ReasoningLevel: Sendable, Equatable {
        case light
        case moderate
        case deep
        case custom(String)
    }

    public var includeSchemaInPrompt: Bool?
    public var reasoningLevel: ReasoningLevel?

    public init(
        includeSchemaInPrompt: Bool? = nil,
        reasoningLevel: ReasoningLevel? = nil
    ) {
        self.includeSchemaInPrompt = includeSchemaInPrompt
        self.reasoningLevel = reasoningLevel
    }
}

extension GenerationOptions {
    public struct ToolCallingMode: Sendable, Equatable {
        public enum Kind: Sendable, Equatable, Hashable {
            case allowed
            case required
            case disallowed
        }

        public var kind: Kind

        private init(kind: Kind) {
            self.kind = kind
        }

        public static let allowed = Self(kind: .allowed)
        public static let required = Self(kind: .required)
        public static let disallowed = Self(kind: .disallowed)
    }
}

extension LanguageModelSession {
    /// A declarative, per-model-request session profile.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26 and mirrors
    ///   Foundation Models 27.
    @_typeEraser(AnyDynamicProfile)
    public protocol DynamicProfile {
        associatedtype Body: DynamicProfile

        @DynamicProfileBuilder
        var body: Body { get }
    }

    /// A concrete profile containing dynamic instructions and Tools.
    public struct Profile: DynamicProfile {
        public typealias Body = Never

        fileprivate let dynamicInstructions: AnyDynamicInstructions
        fileprivate let identity: String

        public init(
            @DynamicInstructionsBuilder _ dynamicInstructions: () -> some DynamicInstructions
        ) {
            let value = dynamicInstructions()
            self.dynamicInstructions = AnyDynamicInstructions(value)
            self.identity = String(reflecting: type(of: value))
        }

        public var body: Never {
            fatalError("LanguageModelSession.Profile has no body")
        }
    }

    /// A type-erased dynamic profile.
    public struct AnyDynamicProfile: DynamicProfile {
        public typealias Body = Never

        fileprivate let resolveValue: () -> ResolvedDynamicProfile

        public init(_ dynamicProfile: any DynamicProfile) {
            resolveValue = { resolveDynamicProfile(dynamicProfile) }
        }

        public init(erasing dynamicProfile: some DynamicProfile) {
            self.init(dynamicProfile)
        }

        public var body: Never {
            fatalError("LanguageModelSession.AnyDynamicProfile has no body")
        }

        func resolveForRequest() -> ResolvedDynamicProfile {
            resolveValue()
        }
    }

    public struct ConditionalDynamicProfile<TrueContent, FalseContent>: DynamicProfile
    where TrueContent: DynamicProfile, FalseContent: DynamicProfile {
        public enum Branch {
            case trueContent(TrueContent)
            case falseContent(FalseContent)
        }

        public typealias Body = Never
        fileprivate let branch: Branch

        public init(_ branch: Branch) {
            self.branch = branch
        }

        public var body: Never {
            fatalError("ConditionalDynamicProfile has no body")
        }
    }

    @resultBuilder
    public struct DynamicProfileBuilder {
        public static func buildBlock<Content>(_ content: Content) -> Content
        where Content: DynamicProfile {
            content
        }

        public static func buildEither<TrueContent, FalseContent>(
            first content: TrueContent
        ) -> ConditionalDynamicProfile<TrueContent, FalseContent>
        where TrueContent: DynamicProfile, FalseContent: DynamicProfile {
            ConditionalDynamicProfile(.trueContent(content))
        }

        public static func buildEither<TrueContent, FalseContent>(
            second content: FalseContent
        ) -> ConditionalDynamicProfile<TrueContent, FalseContent>
        where TrueContent: DynamicProfile, FalseContent: DynamicProfile {
            ConditionalDynamicProfile(.falseContent(content))
        }

        @available(*, unavailable, message: "The body of a 'DynamicProfile' must evaluate to a single active profile")
        public static func buildBlock<each Content>(_ contents: repeat each Content) -> Never
        where repeat each Content: DynamicProfile {
            fatalError("A DynamicProfile must have one active profile")
        }

        public static func buildLimitedAvailability(
            _ component: some DynamicProfile
        ) -> AnyDynamicProfile {
            AnyDynamicProfile(component)
        }
    }

    public protocol DynamicProfileModifier {
        associatedtype Body: DynamicProfile
        typealias Content = DynamicProfileModifierContent<Self>

        @DynamicProfileBuilder
        func body(content: Content) -> Body
    }

    public struct DynamicProfileModifierContent<Modifier>: DynamicProfile
    where Modifier: DynamicProfileModifier {
        public typealias Body = Never
        fileprivate let content: AnyDynamicProfile

        fileprivate init<Content>(_ content: Content) where Content: DynamicProfile {
            self.content = AnyDynamicProfile(content)
        }

        public var body: Never {
            fatalError("DynamicProfileModifierContent has no body")
        }
    }

    public struct ModifiedDynamicProfile<Content, Modifier>: DynamicProfile
    where Content: DynamicProfile, Modifier: DynamicProfileModifier {
        public typealias Body = Never
        fileprivate let content: Content
        fileprivate let modifier: Modifier

        fileprivate init(content: Content, modifier: Modifier) {
            self.content = content
            self.modifier = modifier
        }

        public var body: Never {
            fatalError("ModifiedDynamicProfile has no body")
        }
    }
}

extension LanguageModelSession.DynamicProfile {
    public typealias Profile = LanguageModelSession.Profile
    public typealias DynamicProfile = LanguageModelSession.DynamicProfile
    public typealias SessionProperty = LanguageModelSession.SessionProperty

    public func modifier<Modifier>(
        _ modifier: Modifier
    ) -> some LanguageModelSession.DynamicProfile
    where Modifier: LanguageModelSession.DynamicProfileModifier {
        LanguageModelSession.ModifiedDynamicProfile(content: self, modifier: modifier)
    }

    public func model(
        _ model: any LanguageModel
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .model(model))
    }

    public func model<Model>(
        _ model: Model
    ) -> some LanguageModelSession.DynamicProfile where Model: LanguageModel {
        BuiltinModifiedDynamicProfile(content: self, modifier: .model(model))
    }

    public func temperature(_ temperature: Double?) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .temperature(temperature))
    }

    public func samplingMode(
        _ samplingMode: GenerationOptions.SamplingMode?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .samplingMode(samplingMode))
    }

    public func maximumResponseTokens(
        _ maximumResponseTokens: Int?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(
            content: self,
            modifier: .maximumResponseTokens(maximumResponseTokens)
        )
    }

    public func reasoningLevel(
        _ reasoningLevel: ContextOptions.ReasoningLevel?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .reasoningLevel(reasoningLevel))
    }

    public func toolCallingMode(
        _ toolCallingMode: GenerationOptions.ToolCallingMode?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .toolCallingMode(toolCallingMode))
    }

    public func historyTransform(
        _ transform: @escaping ([Transcript.Entry]) -> [Transcript.Entry]
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .historyTransform(transform))
    }

    public func transcriptErrorHandlingPolicy(
        _ policy: TranscriptErrorHandlingPolicy?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .transcriptErrorHandlingPolicy(policy))
    }

    public func onPrompt(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.Prompt
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onPrompt(action))
    }

    public func onPrompt(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onPrompt { _ in try await action() }
    }

    public func onResponse(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.Response
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onResponse(action))
    }

    public func onResponse(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onResponse { _ in try await action() }
    }

    public func onReasoning(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.Reasoning
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onReasoning(action))
    }

    public func onReasoning(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onReasoning { _ in try await action() }
    }

    public func onToolCall(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.ToolCall
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onToolCall(action))
    }

    public func onToolCall(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onToolCall { _ in try await action() }
    }

    public func onToolOutput(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.ToolCall,
                Transcript.ToolOutput
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onToolOutput(action))
    }

    public func onToolOutput(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onToolOutput { _, _ in try await action() }
    }

    public func onActivate(
        @_inheritActorContext perform action: sending @escaping @isolated(any) () async -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        let wrapped: () async -> Void = { await action() }
        return BuiltinModifiedDynamicProfile(content: self, modifier: .onActivate(wrapped))
    }

    public func onDeactivate(
        @_inheritActorContext perform action: sending @escaping @isolated(any) () async -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        let wrapped: () async -> Void = { await action() }
        return BuiltinModifiedDynamicProfile(content: self, modifier: .onDeactivate(wrapped))
    }
}

extension LanguageModelSession.DynamicProfileModifier {
    public typealias DynamicProfile = LanguageModelSession.DynamicProfile
    public typealias SessionProperty = LanguageModelSession.SessionProperty
}

extension Never: LanguageModelSession.DynamicProfile {}

private protocol PrimitiveDynamicProfile {
    func resolve() -> ResolvedDynamicProfile
}

struct ResolvedDynamicProfile: @unchecked Sendable {
    var identity: String
    var model: (any LanguageModel)?
    var instructions: Instructions?
    var tools: [any Tool]
    var samplingMode: GenerationOptions.SamplingMode?
    var hasSamplingMode = false
    var temperature: Double?
    var hasTemperature = false
    var maximumResponseTokens: Int?
    var hasMaximumResponseTokens = false
    var reasoningLevel: ContextOptions.ReasoningLevel?
    var hasReasoningLevel = false
    var toolCallingMode: GenerationOptions.ToolCallingMode?
    var hasToolCallingMode = false
    var transcriptErrorHandlingPolicy: TranscriptErrorHandlingPolicy?
    var hasTranscriptErrorHandlingPolicy = false
    var historyTransforms: [([Transcript.Entry]) -> [Transcript.Entry]] = []
    var onPrompt: [(Transcript.Prompt) async throws -> Void] = []
    var onResponse: [(Transcript.Response) async throws -> Void] = []
    var onReasoning: [(Transcript.Reasoning) async throws -> Void] = []
    var onToolCall: [(Transcript.ToolCall) async throws -> Void] = []
    var onToolOutput: [(Transcript.ToolCall, Transcript.ToolOutput) async throws -> Void] = []
    var onActivate: [() async -> Void] = []
    var onDeactivate: [() async -> Void] = []
}

extension LanguageModelSession.Profile: PrimitiveDynamicProfile {
    fileprivate func resolve() -> ResolvedDynamicProfile {
        let dynamic = dynamicInstructions.resolveForRequest()
        return ResolvedDynamicProfile(
            identity: identity,
            model: nil,
            instructions: dynamic.instructions,
            tools: dynamic.tools
        )
    }
}

extension LanguageModelSession.AnyDynamicProfile: PrimitiveDynamicProfile {
    fileprivate func resolve() -> ResolvedDynamicProfile {
        resolveForRequest()
    }
}

extension LanguageModelSession.ConditionalDynamicProfile: PrimitiveDynamicProfile {
    fileprivate func resolve() -> ResolvedDynamicProfile {
        var resolved: ResolvedDynamicProfile
        switch branch {
        case .trueContent(let content):
            resolved = resolveDynamicProfile(content)
            resolved.identity = "true/" + resolved.identity
        case .falseContent(let content):
            resolved = resolveDynamicProfile(content)
            resolved.identity = "false/" + resolved.identity
        }
        return resolved
    }
}

extension LanguageModelSession.DynamicProfileModifierContent: PrimitiveDynamicProfile {
    fileprivate func resolve() -> ResolvedDynamicProfile {
        content.resolveForRequest()
    }
}

extension LanguageModelSession.ModifiedDynamicProfile: PrimitiveDynamicProfile {
    fileprivate func resolve() -> ResolvedDynamicProfile {
        resolveDynamicProfile(modifier.body(content: .init(content)))
    }
}

private struct BuiltinModifiedDynamicProfile<Content>: LanguageModelSession.DynamicProfile,
    PrimitiveDynamicProfile
where Content: LanguageModelSession.DynamicProfile {
    typealias Body = Never
    let content: Content
    let modifier: BuiltinDynamicProfileModifier

    var body: Never {
        fatalError("BuiltinModifiedDynamicProfile has no body")
    }

    func resolve() -> ResolvedDynamicProfile {
        var resolved = resolveDynamicProfile(content)
        modifier.apply(to: &resolved)
        return resolved
    }
}

private enum BuiltinDynamicProfileModifier: @unchecked Sendable {
    case model(any LanguageModel)
    case temperature(Double?)
    case samplingMode(GenerationOptions.SamplingMode?)
    case maximumResponseTokens(Int?)
    case reasoningLevel(ContextOptions.ReasoningLevel?)
    case toolCallingMode(GenerationOptions.ToolCallingMode?)
    case historyTransform(([Transcript.Entry]) -> [Transcript.Entry])
    case transcriptErrorHandlingPolicy(TranscriptErrorHandlingPolicy?)
    case onPrompt((Transcript.Prompt) async throws -> Void)
    case onResponse((Transcript.Response) async throws -> Void)
    case onReasoning((Transcript.Reasoning) async throws -> Void)
    case onToolCall((Transcript.ToolCall) async throws -> Void)
    case onToolOutput((Transcript.ToolCall, Transcript.ToolOutput) async throws -> Void)
    case onActivate(() async -> Void)
    case onDeactivate(() async -> Void)

    func apply(to resolved: inout ResolvedDynamicProfile) {
        switch self {
        case .model(let model):
            if resolved.model == nil { resolved.model = model }
        case .temperature(let value):
            if !resolved.hasTemperature {
                resolved.temperature = value
                resolved.hasTemperature = true
            }
        case .samplingMode(let value):
            if !resolved.hasSamplingMode {
                resolved.samplingMode = value
                resolved.hasSamplingMode = true
            }
        case .maximumResponseTokens(let value):
            if !resolved.hasMaximumResponseTokens {
                resolved.maximumResponseTokens = value
                resolved.hasMaximumResponseTokens = true
            }
        case .reasoningLevel(let value):
            if !resolved.hasReasoningLevel {
                resolved.reasoningLevel = value
                resolved.hasReasoningLevel = true
            }
        case .toolCallingMode(let value):
            if !resolved.hasToolCallingMode {
                resolved.toolCallingMode = value
                resolved.hasToolCallingMode = true
            }
        case .transcriptErrorHandlingPolicy(let value):
            if !resolved.hasTranscriptErrorHandlingPolicy {
                resolved.transcriptErrorHandlingPolicy = value
                resolved.hasTranscriptErrorHandlingPolicy = true
            }
        case .historyTransform(let transform):
            resolved.historyTransforms.insert(transform, at: 0)
        case .onPrompt(let action):
            resolved.onPrompt.insert(action, at: 0)
        case .onResponse(let action):
            resolved.onResponse.insert(action, at: 0)
        case .onReasoning(let action):
            resolved.onReasoning.insert(action, at: 0)
        case .onToolCall(let action):
            resolved.onToolCall.insert(action, at: 0)
        case .onToolOutput(let action):
            resolved.onToolOutput.insert(action, at: 0)
        case .onActivate(let action):
            resolved.onActivate.insert(action, at: 0)
        case .onDeactivate(let action):
            resolved.onDeactivate.append(action)
        }
    }
}

private func resolveDynamicProfile(
    _ dynamicProfile: any LanguageModelSession.DynamicProfile
) -> ResolvedDynamicProfile {
    func resolve<Content>(_ content: Content) -> ResolvedDynamicProfile
    where Content: LanguageModelSession.DynamicProfile {
        if let primitive = content as? any PrimitiveDynamicProfile {
            return primitive.resolve()
        }
        return resolve(content.body)
    }

    return resolve(dynamicProfile)
}

actor ProfileLifecycle {
    private var activeProfile: ResolvedDynamicProfile?
    private var lastPromptID: String?

    func prepare(
        _ profile: ResolvedDynamicProfile,
        properties: SessionPropertyValues,
        history: [Transcript.Entry],
        protectedEntryIDs: Set<String> = []
    ) async throws {
        if activeProfile?.identity != profile.identity {
            if let previous = activeProfile {
                await withBindings(
                    properties: properties,
                    history: history,
                    protectedEntryIDs: protectedEntryIDs
                ) {
                    for action in previous.onDeactivate { await action() }
                }
            }
            await withBindings(
                properties: properties,
                history: history,
                protectedEntryIDs: protectedEntryIDs
            ) {
                for action in profile.onActivate { await action() }
            }
            activeProfile = profile
        } else {
            activeProfile = profile
        }

        guard
            let prompt = history.reversed().compactMap({ entry -> Transcript.Prompt? in
                if case .prompt(let prompt) = entry { return prompt }
                return nil
            }).first,
            prompt.id != lastPromptID
        else { return }

        try await withBindings(
            properties: properties,
            history: history,
            protectedEntryIDs: protectedEntryIDs
        ) {
            for action in profile.onPrompt { try await action(prompt) }
        }
        lastPromptID = prompt.id
    }

    private func withBindings<Result: Sendable>(
        properties: SessionPropertyValues,
        history: [Transcript.Entry],
        protectedEntryIDs: Set<String>,
        operation: @Sendable () async throws -> Result
    ) async rethrows -> Result {
        let history = properties.historyBinding(
            history,
            isWritable: true,
            protecting: protectedEntryIDs
        )
        return try await SessionPropertyBinding.$values.withValue(properties) {
            try await SessionPropertyBinding.$history.withValue(history) {
                try await operation()
            }
        }
    }
}

struct MissingDynamicProfileModel: LanguageModel {
    typealias UnavailableReason = Never

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        throw MissingDynamicProfileModelError.missingModelModifier
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            continuation.finish(throwing: MissingDynamicProfileModelError.missingModelModifier)
        }
        return LanguageModelSession.ResponseStream(stream: stream)
    }
}

enum MissingDynamicProfileModelError: Error {
    case missingModelModifier
}
