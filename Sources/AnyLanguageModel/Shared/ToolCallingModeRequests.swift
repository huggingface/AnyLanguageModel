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

extension GenerationOptions {
    /// The tools to offer the model in one request,
    /// for a model that has no tool-choice setting.
    ///
    /// Such a model honors ``ToolCallingMode/disallowed`` by being offered no tools,
    /// so no tool runs.
    /// It can't be made to call a tool,
    /// so it's offered the tools for ``ToolCallingMode/required``
    /// as it is for ``ToolCallingMode/allowed``.
    func toolsOffered(from tools: [any Tool]) -> [any Tool] {
        toolCallingMode?.kind == .disallowed ? [] : tools
    }
}
