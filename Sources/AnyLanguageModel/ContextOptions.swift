/// Options that configure a model's request context.
///
/// This mirrors the Foundation Models 27 request-context surface while keeping
/// the package available on its existing deployment targets.
public struct ContextOptions: Sendable, Equatable {
    /// The amount of reasoning the model should perform for a request.
    public enum ReasoningLevel: Sendable, Equatable {
        case light
        case moderate
        case deep
        case custom(String)
    }

    /// Whether a structured-generation schema should also appear in the prompt.
    public var includeSchemaInPrompt: Bool?

    /// The amount of reasoning the model should perform.
    public var reasoningLevel: ReasoningLevel?

    public init(
        includeSchemaInPrompt: Bool? = nil,
        reasoningLevel: ReasoningLevel? = nil
    ) {
        self.includeSchemaInPrompt = includeSchemaInPrompt
        self.reasoningLevel = reasoningLevel
    }
}
