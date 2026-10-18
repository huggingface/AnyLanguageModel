extension GenerationOptions.ToolCallingMode.Kind {
    /// The `tool_choice` value for this kind
    /// in the OpenAI Chat Completions, OpenAI Responses, and Open Responses APIs.
    var openAIToolChoice: String {
        switch self {
        case .allowed: "auto"
        case .required: "required"
        case .disallowed: "none"
        }
    }
}
