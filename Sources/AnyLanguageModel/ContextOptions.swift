/// Options that configure details that should appear in the prompt.
///
/// No AnyLanguageModel API accepts these options yet.
///
/// - Note: This API is exclusive to AnyLanguageModel on OS 26.
///   It follows the Foundation Models 27 `ContextOptions` API,
///   so code that uses it ports to Foundation Models on OS 27.
public struct ContextOptions: Sendable, Equatable {
    /// A level of reasoning the model is allowed to output before producing a response.
    public enum ReasoningLevel: Sendable, Equatable {
        /// A level that indicates light reasoning that's good for quick responses.
        case light
        /// A level that indicates a moderate amount of reasoning.
        case moderate
        /// A level that indicates deep reasoning that's good for more analysis over a request.
        case deep
        /// A custom level not represented by the other cases.
        case custom(String)
    }

    /// Whether the schema is included in the prompt to bias the model.
    public var includeSchemaInPrompt: Bool?

    /// How much reasoning the model may output before it produces a response.
    public var reasoningLevel: ReasoningLevel?

    /// Creates options that control how the model is prompted.
    ///
    /// - Parameters:
    ///   - includeSchemaInPrompt: Whether the schema is included in the prompt to bias the model.
    ///   - reasoningLevel: How much reasoning the model may output before it produces a response.
    public init(
        includeSchemaInPrompt: Bool? = nil,
        reasoningLevel: ReasoningLevel? = nil
    ) {
        self.includeSchemaInPrompt = includeSchemaInPrompt
        self.reasoningLevel = reasoningLevel
    }
}
