import protocol Foundation.LocalizedError

extension LanguageModelSession {
    /// A declarative description of a session's model, instructions, tools,
    /// options, and callbacks, which the session resolves before each model request.
    ///
    /// Create a session with a profile with ``LanguageModelSession/init(profile:history:)``.
    /// The session evaluates ``body`` again before every model request,
    /// including the requests that continue a response after a tool call,
    /// so the body can choose a different model, instructions, or tools
    /// from the session's properties and history.
    ///
    /// Build a profile from a ``Profile`` and modifiers such as `model(_:)`,
    /// ``temperature(_:)``, and `onResponse(perform:)`.
    /// The body can use `if`-`else` and `if #available` to choose between profiles.
    ///
    /// The profile must set a model with `.model(_:)`.
    /// When a modifier that sets a value appears more than once,
    /// the innermost one, closest to the ``Profile``, sets the value.
    /// A value that you pass when you call a method such as `respond(to:options:)`
    /// overrides the profile's value for that request.
    ///
    /// A session runs its profile's callbacks and lifecycle actions one at a time,
    /// even when requests overlap,
    /// so the state that a callback captures needs no other protection.
    /// While a callback or lifecycle action waits for a request on any session,
    /// as when it awaits `respond(to:)` or consumes a response stream,
    /// the session runs other callbacks and lifecycle actions:
    /// those of the request that it waits for, and those of unrelated requests.
    /// The callback continues only after they end or wait in turn.
    /// So, as with actor reentrancy, captured state can change across such a wait,
    /// but two callbacks never run at the same time.
    /// A request that the callback makes in another task,
    /// such as an unstructured task that it doesn't wait for,
    /// runs its callbacks while the first callback waits for a request or a response stream,
    /// or after the first callback returns.
    /// So a callback that waits for another task
    /// that makes a request on the same session never finishes.
    /// This includes a child task, from `async let` or a task group,
    /// and an unstructured or detached task that the callback awaits with `Task.value`.
    /// History transforms also run one at a time.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.DynamicProfile` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    ///   Unlike Foundation Models, which uses the system language model
    ///   when a profile sets no model,
    ///   AnyLanguageModel requires a model.
    @_typeEraser(AnyDynamicProfile)
    public protocol DynamicProfile {
        /// The type of profile that the body produces.
        associatedtype Body: DynamicProfile

        /// The profile that this profile resolves to.
        ///
        /// The session evaluates the body before each model request.
        @DynamicProfileBuilder
        var body: Body { get }
    }

    /// A profile with dynamic instructions and tools,
    /// which modifiers add a model, options, and callbacks to.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.Profile` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    public struct Profile: DynamicProfile {
        /// A profile has no body; the session reads its instructions and tools directly.
        public typealias Body = Never

        fileprivate let dynamicInstructions: AnyDynamicInstructions
        fileprivate let identity: String

        /// Creates a profile with instructions and tools that you build with a result builder.
        ///
        /// The session evaluates `dynamicInstructions` before each model request.
        ///
        /// - Parameter dynamicInstructions: A closure that returns the instructions and tools.
        public init(
            @DynamicInstructionsBuilder _ dynamicInstructions: () -> some DynamicInstructions
        ) {
            let value = dynamicInstructions()
            self.dynamicInstructions = AnyDynamicInstructions(value)
            self.identity = String(reflecting: type(of: value))
        }

        /// A profile has no body.
        ///
        /// Accessing this property stops the program.
        public var body: Never {
            fatalError("LanguageModelSession.Profile has no body")
        }
    }

    /// A type-erased dynamic profile.
    ///
    /// The ``DynamicProfileBuilder`` uses this type for an `if #available` branch.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.AnyDynamicProfile` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    public struct AnyDynamicProfile: DynamicProfile {
        /// A type-erased profile has no body; the session resolves the profile that it wraps.
        public typealias Body = Never

        fileprivate let resolveValue: () -> ResolvedDynamicProfile

        /// Creates a type-erased profile that wraps the given profile.
        ///
        /// - Parameter dynamicProfile: The profile to wrap.
        public init(_ dynamicProfile: any DynamicProfile) {
            let identity = String(reflecting: type(of: dynamicProfile))
            resolveValue = {
                var resolved = resolveDynamicProfile(dynamicProfile)
                resolved.identity = identity + "/" + resolved.identity
                return resolved
            }
        }

        /// Creates a type-erased profile that wraps the given profile.
        ///
        /// - Parameter dynamicProfile: The profile to wrap.
        public init(erasing dynamicProfile: some DynamicProfile) {
            self.init(dynamicProfile)
        }

        /// A type-erased profile has no body.
        ///
        /// Accessing this property stops the program.
        public var body: Never {
            fatalError("LanguageModelSession.AnyDynamicProfile has no body")
        }

        func resolveForRequest() -> ResolvedDynamicProfile {
            resolveValue()
        }
    }

    /// A profile that the ``DynamicProfileBuilder`` creates
    /// for the branches of an `if`-`else` statement.
    ///
    /// The session treats the two branches as different profiles,
    /// so a change of branch between requests runs the lifecycle actions
    /// of ``DynamicProfile/onDeactivate(perform:)`` and ``DynamicProfile/onActivate(perform:)``.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.ConditionalDynamicProfile` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    public struct ConditionalDynamicProfile<TrueContent, FalseContent>: DynamicProfile
    where TrueContent: DynamicProfile, FalseContent: DynamicProfile {
        enum Branch {
            case trueContent(TrueContent)
            case falseContent(FalseContent)
        }

        /// A conditional profile has no body; the session resolves the active branch.
        public typealias Body = Never
        fileprivate let branch: Branch

        init(_ branch: Branch) {
            self.branch = branch
        }

        /// A conditional profile has no body.
        ///
        /// Accessing this property stops the program.
        public var body: Never {
            fatalError("ConditionalDynamicProfile has no body")
        }
    }

    /// A result builder that creates a dynamic profile from a closure.
    ///
    /// The body must produce one profile.
    /// It can use `if`-`else` to choose between two profiles,
    /// and `if #available` with an `else` branch to use a profile
    /// only on some operating system versions.
    /// An `if` statement without an `else` branch isn't supported.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.DynamicProfileBuilder` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    @resultBuilder
    public struct DynamicProfileBuilder {
        /// Returns the single profile in a body unchanged.
        public static func buildBlock<Content>(_ content: Content) -> Content
        where Content: DynamicProfile {
            content
        }

        /// Returns a profile for the first branch of an `if`-`else` statement.
        public static func buildEither<TrueContent, FalseContent>(
            first content: TrueContent
        ) -> ConditionalDynamicProfile<TrueContent, FalseContent>
        where TrueContent: DynamicProfile, FalseContent: DynamicProfile {
            ConditionalDynamicProfile(.trueContent(content))
        }

        /// Returns a profile for the second branch of an `if`-`else` statement.
        public static func buildEither<TrueContent, FalseContent>(
            second content: FalseContent
        ) -> ConditionalDynamicProfile<TrueContent, FalseContent>
        where TrueContent: DynamicProfile, FalseContent: DynamicProfile {
            ConditionalDynamicProfile(.falseContent(content))
        }

        /// A body with more than one profile doesn't compile;
        /// combine profiles with `if`-`else` or modifiers instead.
        @available(*, unavailable, message: "The body of a 'DynamicProfile' must evaluate to a single active profile")
        public static func buildBlock<each Content>(_ contents: repeat each Content) -> Never
        where repeat each Content: DynamicProfile {
            fatalError("A DynamicProfile must have one active profile")
        }

        /// Returns a type-erased profile for an `if #available` branch.
        public static func buildLimitedAvailability(
            _ component: some DynamicProfile
        ) -> AnyDynamicProfile {
            AnyDynamicProfile(component)
        }
    }

    /// A modifier that you apply to a dynamic profile
    /// to produce a different version of the profile.
    ///
    /// Apply a modifier with ``DynamicProfile/modifier(_:)``.
    /// The modifiers that ``body(content:)`` applies to its content
    /// are outer to the modifiers in the content,
    /// so the content's own values take precedence
    /// and the modifier's callbacks run before the content's,
    /// as for any other modifier.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.DynamicProfileModifier` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    public protocol DynamicProfileModifier {
        /// The type of profile that the modifier produces.
        associatedtype Body: DynamicProfile

        /// The profile that the modifier receives as its content.
        typealias Content = DynamicProfileModifierContent<Self>

        /// Returns the modified profile.
        ///
        /// The session calls this method before each model request.
        ///
        /// - Parameter content: The profile that the modifier applies to.
        @DynamicProfileBuilder
        func body(content: Content) -> Body
    }

    /// The profile that a ``DynamicProfileModifier`` receives as its content.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.DynamicProfileModifierContent` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    public struct DynamicProfileModifierContent<Modifier>: DynamicProfile
    where Modifier: DynamicProfileModifier {
        /// The content has no body; the session resolves the profile that it wraps.
        public typealias Body = Never
        fileprivate let content: AnyDynamicProfile

        fileprivate init<Content>(_ content: Content) where Content: DynamicProfile {
            self.content = AnyDynamicProfile(content)
        }

        /// The content has no body.
        ///
        /// Accessing this property stops the program.
        public var body: Never {
            fatalError("DynamicProfileModifierContent has no body")
        }
    }

    /// A profile with a ``DynamicProfileModifier`` applied to it.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26.
    ///   It follows the Foundation Models 27 `LanguageModelSession.ModifiedDynamicProfile` API,
    ///   so code that uses it ports to Foundation Models on OS 27.
    public struct ModifiedDynamicProfile<Content, Modifier>: DynamicProfile
    where Content: DynamicProfile, Modifier: DynamicProfileModifier {
        /// A modified profile has no body; the session resolves the modifier's body.
        public typealias Body = Never
        fileprivate let content: Content
        fileprivate let modifier: Modifier

        fileprivate init(content: Content, modifier: Modifier) {
            self.content = content
            self.modifier = modifier
        }

        /// A modified profile has no body.
        ///
        /// Accessing this property stops the program.
        public var body: Never {
            fatalError("ModifiedDynamicProfile has no body")
        }
    }
}

// Modifiers that set a value: when a modifier repeats, the innermost one wins
// (`BuiltinDynamicProfileModifier.apply` keeps a value that the content set).
// Callback modifiers and history transforms: the outermost runs first,
// except `onDeactivate`, where the innermost runs first.
extension LanguageModelSession.DynamicProfile {
    /// A profile with dynamic instructions and tools.
    ///
    /// This alias lets a profile's body write `Profile` instead of `LanguageModelSession.Profile`.
    public typealias Profile = LanguageModelSession.Profile

    /// A declarative description of a session's model, instructions, tools, and callbacks.
    ///
    /// This alias lets a profile's body write `DynamicProfile`
    /// instead of `LanguageModelSession.DynamicProfile`.
    public typealias DynamicProfile = LanguageModelSession.DynamicProfile

    /// A property wrapper that reads and writes a session property.
    ///
    /// This alias lets a profile write `@SessionProperty`
    /// instead of `@LanguageModelSession.SessionProperty`.
    public typealias SessionProperty = LanguageModelSession.SessionProperty

    /// Applies a custom modifier to this profile.
    ///
    /// - Parameter modifier: The modifier to apply.
    /// - Returns: A profile with the modifier applied.
    public func modifier<Modifier>(
        _ modifier: Modifier
    ) -> some LanguageModelSession.DynamicProfile
    where Modifier: LanguageModelSession.DynamicProfileModifier {
        LanguageModelSession.ModifiedDynamicProfile(content: self, modifier: modifier)
    }

    /// Sets the language model that responds to requests with this profile.
    ///
    /// A profile must set a model.
    /// If the active profile sets none, a request throws
    /// ``LanguageModelSession/DynamicProfileError/missingModel``
    /// before it changes the transcript or runs any of the profile's actions.
    /// If the modifier appears more than once, the innermost one sets the model.
    ///
    /// - Parameter model: The model to use.
    /// - Returns: A profile that uses the model.
    ///
    /// - Note: Foundation Models takes its own `LanguageModel` type here
    ///   and uses the system language model when a profile sets no model.
    ///   AnyLanguageModel takes an AnyLanguageModel ``LanguageModel`` and requires a model.
    public func model(
        _ model: any LanguageModel
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .model(model))
    }

    /// Sets the language model that responds to requests with this profile.
    ///
    /// A profile must set a model.
    /// If the active profile sets none, a request throws
    /// ``LanguageModelSession/DynamicProfileError/missingModel``
    /// before it changes the transcript or runs any of the profile's actions.
    /// If the modifier appears more than once, the innermost one sets the model.
    ///
    /// - Parameter model: The model to use.
    /// - Returns: A profile that uses the model.
    ///
    /// - Note: Foundation Models takes its own `LanguageModel` type here
    ///   and uses the system language model when a profile sets no model.
    ///   AnyLanguageModel takes an AnyLanguageModel ``LanguageModel`` and requires a model.
    public func model<Model>(
        _ model: Model
    ) -> some LanguageModelSession.DynamicProfile where Model: LanguageModel {
        BuiltinModifiedDynamicProfile(content: self, modifier: .model(model))
    }

    /// Sets the default temperature for requests with this profile.
    ///
    /// A non-`nil` ``GenerationOptions/temperature`` that you pass with a request
    /// overrides this value.
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter temperature: The temperature, or `nil` for the model's default.
    /// - Returns: A profile with the temperature.
    public func temperature(_ temperature: Double?) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .temperature(temperature))
    }

    /// Sets the default sampling mode for requests with this profile.
    ///
    /// A non-`nil` ``GenerationOptions/sampling`` that you pass with a request
    /// overrides this value.
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter samplingMode: The sampling mode, or `nil` for the model's default.
    /// - Returns: A profile with the sampling mode.
    public func samplingMode(
        _ samplingMode: GenerationOptions.SamplingMode?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .samplingMode(samplingMode))
    }

    /// Sets the default maximum number of tokens in a response for requests with this profile.
    ///
    /// A non-`nil` ``GenerationOptions/maximumResponseTokens`` that you pass with a request
    /// overrides this value.
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter maximumResponseTokens: The maximum number of tokens,
    ///   or `nil` for the model's default.
    /// - Returns: A profile with the maximum.
    public func maximumResponseTokens(
        _ maximumResponseTokens: Int?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(
            content: self,
            modifier: .maximumResponseTokens(maximumResponseTokens)
        )
    }

    /// Sets how much the model may reason before it responds to requests with this profile.
    ///
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter reasoningLevel: The reasoning level, or `nil` for the model's default.
    /// - Returns: A profile with the reasoning level.
    ///
    /// - Important: At this time, none of the language models in AnyLanguageModel
    ///   apply this value.
    public func reasoningLevel(
        _ reasoningLevel: ContextOptions.ReasoningLevel?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .reasoningLevel(reasoningLevel))
    }

    /// Sets whether requests with this profile include the response schema in the prompt.
    ///
    /// Unlike the options that ``GenerationOptions`` carries,
    /// a non-`nil` value here overrides the `includeSchemaInPrompt` argument
    /// that you pass with a request.
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter includeSchemaInPrompt: Whether to include the schema,
    ///   or `nil` to use the request's argument.
    /// - Returns: A profile with the setting.
    @_spi(Compatibility)
    public func includeSchemaInPrompt(
        _ includeSchemaInPrompt: Bool?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(
            content: self,
            modifier: .includeSchemaInPrompt(includeSchemaInPrompt)
        )
    }

    /// Sets the default tool calling mode for requests with this profile.
    ///
    /// A non-`nil` ``GenerationOptions/toolCallingMode`` that you pass with a request
    /// overrides this value.
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter toolCallingMode: The tool calling mode, or `nil` for the model's default.
    /// - Returns: A profile with the tool calling mode.
    public func toolCallingMode(
        _ toolCallingMode: GenerationOptions.ToolCallingMode?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .toolCallingMode(toolCallingMode))
    }

    /// Changes the history that the model receives with each request.
    ///
    /// Before each model request, the session calls `transform` with the session history
    /// and sends the entries that it returns to the model in place of the history.
    /// The session's ``LanguageModelSession/transcript`` doesn't change,
    /// unless the transform writes to the `history` session property.
    ///
    /// When a profile has more than one history transform, the outermost one runs first,
    /// and each one receives the entries that the one before it returned.
    /// Transforms run synchronously and one at a time, while the session holds a lock.
    ///
    /// - Parameter transform: A closure that returns the entries to send to the model.
    /// - Returns: A profile with the history transform.
    public func historyTransform(
        _ transform: @escaping ([Transcript.Entry]) -> [Transcript.Entry]
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .historyTransform(transform))
    }

    /// Sets what the transcript keeps when a request with this profile fails.
    ///
    /// The value replaces the session's
    /// ``LanguageModelSession/transcriptErrorHandlingPolicy`` for the request,
    /// including when the value is `nil`.
    /// If the modifier appears more than once, the innermost one sets the value.
    ///
    /// - Parameter policy: The policy for a failed request.
    /// - Returns: A profile with the policy.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel
    ///   and using it means your code is no longer drop-in compatible
    ///   with the Foundation Models framework.
    public func transcriptErrorHandlingPolicy(
        _ policy: TranscriptErrorHandlingPolicy?
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .transcriptErrorHandlingPolicy(policy))
    }

    /// Adds an action to run when the session receives a prompt.
    ///
    /// The action runs once for each prompt, after the session adds the prompt to the transcript
    /// and before the first model request for it,
    /// and after the lifecycle actions of ``onActivate(perform:)``.
    /// If the action throws, the request throws the error.
    /// When a profile has more than one `onPrompt` action, the outermost one runs first.
    ///
    /// The action can't make a request on its own session,
    /// directly or through a request on another session whose actions make one;
    /// that request throws ``LanguageModelSession/DynamicProfileError/requestDuringLifecycleAction``.
    ///
    /// - Parameter action: The action to run, which receives the prompt.
    /// - Returns: A profile with the action.
    public func onPrompt(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.Prompt
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onPrompt(action))
    }

    /// Adds an action to run when the session receives a prompt.
    ///
    /// This is the same as `onPrompt(perform:)`
    /// for an action that doesn't use the prompt.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    public func onPrompt(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onPrompt { _ in try await action() }
    }

    /// Adds an action to run when the model produces a response.
    ///
    /// The action runs once for each response,
    /// after the session adds the response to the transcript;
    /// for a streamed response, it runs after the stream ends.
    /// If the action throws, the session removes the response from the transcript,
    /// and the request throws the error.
    /// When a profile has more than one `onResponse` action, the outermost one runs first.
    ///
    /// - Parameter action: The action to run, which receives the response.
    /// - Returns: A profile with the action.
    public func onResponse(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.Response
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onResponse(action))
    }

    /// Adds an action to run when the model produces a response.
    ///
    /// This is the same as `onResponse(perform:)`
    /// for an action that doesn't use the response.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    public func onResponse(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onResponse { _ in try await action() }
    }

    /// Adds an action to run when the model produces reasoning.
    ///
    /// The action runs once for each reasoning entry in a response,
    /// after the session adds the response to the transcript
    /// and before the actions of `onResponse(perform:)`.
    /// If the action throws, the session removes the response from the transcript,
    /// and the request throws the error.
    /// When a profile has more than one `onReasoning` action, the outermost one runs first.
    ///
    /// - Parameter action: The action to run, which receives the reasoning.
    /// - Returns: A profile with the action.
    public func onReasoning(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.Reasoning
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onReasoning(action))
    }

    /// Adds an action to run when the model produces reasoning.
    ///
    /// This is the same as `onReasoning(perform:)`
    /// for an action that doesn't use the reasoning.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    public func onReasoning(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onReasoning { _ in try await action() }
    }

    /// Adds an action to run before the session calls a tool.
    ///
    /// When a profile has more than one `onToolCall` action, the outermost one runs first.
    ///
    /// - Parameter action: The action to run, which receives the tool call.
    /// - Returns: A profile with the action.
    ///
    /// - Important: At this time, none of the language models in AnyLanguageModel
    ///   run this action when they call a tool.
    public func onToolCall(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.ToolCall
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onToolCall(action))
    }

    /// Adds an action to run before the session calls a tool.
    ///
    /// This is the same as `onToolCall(perform:)`
    /// for an action that doesn't use the tool call.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    ///
    /// - Important: At this time, none of the language models in AnyLanguageModel
    ///   run this action when they call a tool.
    public func onToolCall(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onToolCall { _ in try await action() }
    }

    /// Adds an action to run after a tool produces its output.
    ///
    /// When a profile has more than one `onToolOutput` action, the outermost one runs first.
    ///
    /// - Parameter action: The action to run, which receives the tool call and its output.
    /// - Returns: A profile with the action.
    ///
    /// - Important: At this time, none of the language models in AnyLanguageModel
    ///   run this action when they call a tool.
    public func onToolOutput(
        @_inheritActorContext perform action:
            nonisolated(nonsending) sending @escaping (
                Transcript.ToolCall,
                Transcript.ToolOutput
            ) async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        BuiltinModifiedDynamicProfile(content: self, modifier: .onToolOutput(action))
    }

    /// Adds an action to run after a tool produces its output.
    ///
    /// This is the same as `onToolOutput(perform:)`
    /// for an action that doesn't use the tool call or its output.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    ///
    /// - Important: At this time, none of the language models in AnyLanguageModel
    ///   run this action when they call a tool.
    public func onToolOutput(
        @_inheritActorContext perform action: nonisolated(nonsending) sending @escaping () async throws -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        onToolOutput { _, _ in try await action() }
    }

    /// Adds an action to run when this profile becomes the session's active profile.
    ///
    /// Before a model request, the session compares the profile that the body resolves to
    /// with the profile of the previous request.
    /// For the first request, or when the profile changed,
    /// such as when the body takes a different `if`-`else` branch,
    /// the session runs the previous profile's ``onDeactivate(perform:)`` actions
    /// and then the new profile's `onActivate` actions,
    /// before the actions of `onPrompt(perform:)`.
    /// When a profile has more than one `onActivate` action, the outermost one runs first.
    ///
    /// The action can't make a request on its own session,
    /// directly or through a request on another session whose actions make one;
    /// that request throws ``LanguageModelSession/DynamicProfileError/requestDuringLifecycleAction``.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    public func onActivate(
        @_inheritActorContext perform action: sending @escaping @isolated(any) () async -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        let wrapped: () async -> Void = { await action() }
        return BuiltinModifiedDynamicProfile(content: self, modifier: .onActivate(wrapped))
    }

    /// Adds an action to run when this profile stops being the session's active profile.
    ///
    /// The action runs before a model request whose profile differs from this one,
    /// before the new profile's ``onActivate(perform:)`` actions.
    /// It doesn't run when the session ends.
    /// When a profile has more than one `onDeactivate` action, the innermost one runs first.
    ///
    /// The action can't make a request on its own session,
    /// directly or through a request on another session whose actions make one;
    /// that request throws ``LanguageModelSession/DynamicProfileError/requestDuringLifecycleAction``.
    ///
    /// - Parameter action: The action to run.
    /// - Returns: A profile with the action.
    public func onDeactivate(
        @_inheritActorContext perform action: sending @escaping @isolated(any) () async -> Void
    ) -> some LanguageModelSession.DynamicProfile {
        let wrapped: () async -> Void = { await action() }
        return BuiltinModifiedDynamicProfile(content: self, modifier: .onDeactivate(wrapped))
    }
}

extension LanguageModelSession.DynamicProfileModifier {
    /// A declarative description of a session's model, instructions, tools, and callbacks.
    ///
    /// This alias lets a modifier write `DynamicProfile`
    /// instead of `LanguageModelSession.DynamicProfile`.
    public typealias DynamicProfile = LanguageModelSession.DynamicProfile

    /// A property wrapper that reads and writes a session property.
    ///
    /// This alias lets a modifier write `@SessionProperty`
    /// instead of `@LanguageModelSession.SessionProperty`.
    public typealias SessionProperty = LanguageModelSession.SessionProperty
}

/// `Never` is the body type of profiles that have no body,
/// such as ``LanguageModelSession/Profile``.
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
    var includeSchemaInPrompt: Bool?
    var hasIncludeSchemaInPrompt = false
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
        let history = SessionPropertyBinding.history.map {
            SessionHistoryBinding($0.entries, isWritable: false)
        }
        let dynamic = SessionPropertyBinding.$history.withValue(history) {
            dynamicInstructions.resolveForRequest()
        }
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
    case includeSchemaInPrompt(Bool?)
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
        case .includeSchemaInPrompt(let value):
            if !resolved.hasIncludeSchemaInPrompt {
                resolved.includeSchemaInPrompt = value
                resolved.hasIncludeSchemaInPrompt = true
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
    /// One queued or running preparation of a session's profile for a request,
    /// which runs the profile's lifecycle actions.
    ///
    /// A task that the actions start inherits the preparation,
    /// but the preparation stops counting when it ends,
    /// so that task's later requests don't fail.
    ///
    /// Preparations also record which preparations each one waits for:
    /// its predecessor in the session's queue,
    /// and the preparations of the requests that its lifecycle actions make.
    /// A request whose preparation would close a cycle of these waits,
    /// such as when session A's action waits for a request on session B
    /// while B's action, from another request, waits for a request on A,
    /// throws ``LanguageModelSession/DynamicProfileError/requestDuringLifecycleAction``
    /// instead of waiting forever.
    final class Preparation: Sendable {
        /// The preparations that each preparation waits for, by identifier.
        ///
        /// A preparation stays alive while a wait for it or by it is recorded,
        /// so its identifier isn't reused for another one in that time.
        private static let waits = Locked<[ObjectIdentifier: [ObjectIdentifier]]>([:])

        /// The lifecycle that runs the actions.
        let lifecycle: ObjectIdentifier
        private let state = Locked(true)

        init(lifecycle: ObjectIdentifier) {
            self.lifecycle = lifecycle
        }

        /// Whether the preparation is queued or runs its lifecycle actions.
        var isRunning: Bool { state.withLock { $0 } }

        /// Records that the preparation ended.
        func end() { state.withLock { $0 = false } }

        /// Records that `waiting` wait for this preparation,
        /// which waits for `predecessor`.
        ///
        /// Throws ``LanguageModelSession/DynamicProfileError/requestDuringLifecycleAction``
        /// and records nothing if `predecessor` waits for any of `waiting`,
        /// directly or through other preparations.
        func beginWaits(by waiting: [Preparation], after predecessor: Preparation?) throws {
            let waitingIDs = waiting.map(ObjectIdentifier.init)
            let id = ObjectIdentifier(self)
            try Self.waits.withLock { waits in
                if let predecessor {
                    let predecessorID = ObjectIdentifier(predecessor)
                    if Self.path(in: waits, from: predecessorID, toAnyOf: Set(waitingIDs)) {
                        throw LanguageModelSession.DynamicProfileError.requestDuringLifecycleAction
                    }
                    waits[id, default: []].append(predecessorID)
                }
                for waitingID in waitingIDs {
                    waits[waitingID, default: []].append(id)
                }
            }
        }

        /// Records that this preparation no longer waits for `predecessor`.
        func endWait(for predecessor: Preparation) {
            Self.removeWait(by: ObjectIdentifier(self), for: ObjectIdentifier(predecessor))
        }

        /// Records that `waiting` no longer wait for this preparation.
        func endWaits(by waiting: [Preparation]) {
            for preparation in waiting {
                Self.removeWait(by: ObjectIdentifier(preparation), for: ObjectIdentifier(self))
            }
        }

        private static func removeWait(by waiting: ObjectIdentifier, for preparation: ObjectIdentifier) {
            waits.withLock { waits in
                guard var targets = waits[waiting],
                    let index = targets.firstIndex(of: preparation)
                else { return }
                targets.remove(at: index)
                waits[waiting] = targets.isEmpty ? nil : targets
            }
        }

        private static func path(
            in waits: [ObjectIdentifier: [ObjectIdentifier]],
            from start: ObjectIdentifier,
            toAnyOf targets: Set<ObjectIdentifier>
        ) -> Bool {
            var visited: Set<ObjectIdentifier> = []
            var pending = [start]
            while let next = pending.popLast() {
                if targets.contains(next) { return true }
                guard visited.insert(next).inserted else { continue }
                pending.append(contentsOf: waits[next] ?? [])
            }
            return false
        }
    }

    /// The preparations whose lifecycle actions run the current task's work.
    ///
    /// A lifecycle action can make a request on another session,
    /// whose lifecycle action can make a request on a third session, and so on.
    /// Each preparation adds itself to the running ones that it inherits.
    @TaskLocal private static var preparations: [Preparation] = []

    /// The serializer that runs the session's profile callbacks and lifecycle actions one at a time.
    nonisolated let actionSerializer = ProfileActionSerializer()

    private var activeProfile: ResolvedDynamicProfile?
    private var pendingPreparation: (preparation: Preparation, task: Task<Void, any Error>)?

    func prepare(
        _ profile: ResolvedDynamicProfile,
        properties: SessionPropertyValues,
        history: [Transcript.Entry],
        prompt: Transcript.Prompt?,
        callbackState: ProfileCallbackState,
        historyEdits: Locked<[Transcript.Entry]?>,
        historyVersion: Locked<UInt64?>,
        protectedEntryKeys: Set<TranscriptEntryKey> = []
    ) async throws {
        // A lifecycle action that awaits a request on the same session,
        // directly or through requests on other sessions,
        // would wait for its own preparation to finish.
        let waiting = Self.preparations.filter(\.isRunning)
        guard !waiting.contains(where: { $0.lifecycle == ObjectIdentifier(self) }) else {
            throw LanguageModelSession.DynamicProfileError.requestDuringLifecycleAction
        }
        // Run no lifecycle action for a profile that can't make the request.
        guard profile.model != nil else {
            throw LanguageModelSession.DynamicProfileError.missingModel
        }
        let previous = pendingPreparation
        let preparation = Preparation(lifecycle: ObjectIdentifier(self))
        try preparation.beginWaits(by: waiting, after: previous?.preparation)
        defer { preparation.endWaits(by: waiting) }
        let task = Task {
            defer { preparation.end() }
            if let previous {
                // Wait for the previous preparation without cancelling it.
                _ = try? await previous.task.value
                preparation.endWait(for: previous.preparation)
            }
            try Task.checkCancellation()
            // Wait for the permit after the previous preparation,
            // which may need the permit to finish.
            try await actionSerializer.run {
                try await Self.$preparations.withValue(waiting + [preparation]) {
                    try await self.performPreparation(
                        profile,
                        properties: properties,
                        history: history,
                        prompt: prompt,
                        callbackState: callbackState,
                        historyEdits: historyEdits,
                        historyVersion: historyVersion,
                        protectedEntryKeys: protectedEntryKeys
                    )
                }
            }
        }
        pendingPreparation = (preparation, task)
        defer { if pendingPreparation?.preparation === preparation { pendingPreparation = nil } }
        // The preparation runs in an unstructured task so that it stays ordered
        // after its predecessor; forward the request's cancellation to it.
        // A callback that waits for it lets the preparation and other callbacks run.
        try await ProfileActionSerializer.withPermitsReleased {
            try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
        }
    }

    private func performPreparation(
        _ profile: ResolvedDynamicProfile,
        properties: SessionPropertyValues,
        history: [Transcript.Entry],
        prompt: Transcript.Prompt?,
        callbackState: ProfileCallbackState,
        historyEdits: Locked<[Transcript.Entry]?>,
        historyVersion: Locked<UInt64?>,
        protectedEntryKeys: Set<TranscriptEntryKey>
    ) async throws {
        let history = properties.historyBinding(
            history,
            isWritable: true,
            protecting: protectedEntryKeys,
            persistingAt: historyVersion,
            onChange: { updated in historyEdits.withLock { $0 = updated } }
        )
        try Task.checkCancellation()
        if activeProfile?.identity != profile.identity {
            if let previous = activeProfile {
                await withBindings(
                    properties: properties,
                    history: history
                ) {
                    for action in previous.onDeactivate { await action() }
                }
            }
            await withBindings(
                properties: properties,
                history: history
            ) {
                for action in profile.onActivate { await action() }
            }
            activeProfile = profile
            // The hooks ran, so the profile is active even if their history edits weren't kept.
            if history.hasConflict {
                throw LanguageModelSession.DynamicProfileError.historyMutationWhileRunningCallback
            }
        } else {
            activeProfile = profile
        }

        guard let prompt else { return }
        try Task.checkCancellation()
        guard callbackState.reserve(prompt.id) else { return }
        do {
            try await withBindings(
                properties: properties,
                history: history
            ) {
                for action in profile.onPrompt { try await action(prompt) }
            }
            if history.hasConflict {
                throw LanguageModelSession.DynamicProfileError.historyMutationWhileRunningCallback
            }
        } catch {
            callbackState.release(prompt.id)
            throw error
        }
    }

    private func withBindings<Result: Sendable>(
        properties: SessionPropertyValues,
        history: SessionHistoryBinding,
        operation: @Sendable () async throws -> Result
    ) async rethrows -> Result {
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
        throw LanguageModelSession.DynamicProfileError.missingModel
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
            continuation.finish(throwing: LanguageModelSession.DynamicProfileError.missingModel)
        }
        return LanguageModelSession.ResponseStream(stream: stream)
    }
}

extension LanguageModelSession {
    /// An error that occurs when a dynamic profile can't make a request.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel.
    ///   Foundation Models uses the system language model
    ///   when a profile sets no model.
    public enum DynamicProfileError: Error, LocalizedError, Sendable, Equatable {
        /// The active profile sets no model.
        case missingModel

        /// The session history changed while the active profile edited it.
        ///
        /// Profile resolution reads one snapshot of the history.
        /// If the profile edits that history,
        /// the session replaces its history with the edited history
        /// only if no other change happened since the snapshot,
        /// such as a response from an overlapping request
        /// or a direct `properties.history` write.
        /// Otherwise, the session keeps the other change and throws this error.
        /// It doesn't keep the profile's edits, and a new request's prompt isn't added.
        /// Unlike Foundation Models,
        /// which rejects a transcript change while a session responds,
        /// AnyLanguageModel allows overlapping requests.
        case historyMutationWhileResolving

        /// The session history changed before a profile callback edited it.
        ///
        /// A profile's lifecycle actions and its callbacks,
        /// such as `onToolCall`, `onToolOutput`, `onReasoning`, and `onResponse`,
        /// see the history from the request's latest snapshot.
        /// The session keeps a callback's history edit only if
        /// no other change happened since that snapshot,
        /// such as a response from an overlapping request
        /// or a direct `properties.history` write.
        /// Otherwise, the session keeps the other change
        /// and throws this error after the callback returns.
        /// It doesn't keep that edit or any later edit in the request,
        /// and the request fails like a request whose callback throws.
        case historyMutationWhileRunningCallback

        /// A profile's lifecycle action made a request that would wait for the action.
        ///
        /// A session prepares one request's profile at a time,
        /// so a request on the same session that an `onActivate`, `onDeactivate`,
        /// or `onPrompt` action awaits would wait for that action to finish.
        /// The session throws this error from the nested request instead.
        /// It also throws this error when the request would wait for the action
        /// through other sessions,
        /// as when session A's action makes a request on session B,
        /// and B's action makes a request on A.
        /// When those actions run for separate requests,
        /// the request that would complete the cycle throws this error,
        /// and the other request continues.
        /// The session can detect only a request that the action makes
        /// in its own task or in a child or unstructured task of it,
        /// not one in a detached task.
        /// A request from a task that the action starts
        /// throws this error only while the action runs;
        /// after the action ends, the request runs normally.
        case requestDuringLifecycleAction

        /// A message that describes the error.
        public var errorDescription: String? {
            switch self {
            case .missingModel:
                return "The active dynamic profile sets no model. Add `.model(_:)` to the profile."
            case .historyMutationWhileResolving:
                return
                    "The session history changed while the active dynamic profile edited it. The profile's edits weren't kept."
            case .historyMutationWhileRunningCallback:
                return
                    "The session history changed before a dynamic profile callback edited it. The callback's edits weren't kept."
            case .requestDuringLifecycleAction:
                return
                    "A dynamic profile's lifecycle action made a request that would wait for the action to finish."
            }
        }
    }
}
