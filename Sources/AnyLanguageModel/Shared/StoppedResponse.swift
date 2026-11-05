extension LanguageModel {
    /// The response that ends when a ``ToolExecutionDelegate`` stops a tool call
    /// before the model writes an answer.
    ///
    /// A `String`, or a type that can be empty, ends with an empty value.
    /// A type with required properties can't be empty,
    /// so this throws ``LanguageModelSession/ToolExecutionStopped`` with the entries instead.
    func stoppedResponse<Content: Generable>(
        generating type: Content.Type,
        provider: String,
        transcriptEntries: [Transcript.Entry],
        requests: RequestsUsage
    ) throws -> LanguageModelSession.Response<Content> {
        func response(_ content: Content, _ rawContent: GeneratedContent) -> LanguageModelSession.Response<Content> {
            .init(
                content: content,
                rawContent: rawContent,
                transcriptEntries: ArraySlice(transcriptEntries),
                requests: requests
            )
        }
        if type == String.self {
            return response("" as! Content, GeneratedContent(""))
        }
        let emptyObject = GeneratedContent(properties: [:])
        if let content = try? Content(emptyObject) {
            return response(content, emptyObject)
        }
        let nullContent = GeneratedContent(kind: .null)
        if let content = try? Content(nullContent) {
            return response(content, nullContent)
        }
        throw LanguageModelSession.ToolExecutionStopped(
            provider: provider,
            transcriptEntries: ArraySlice(transcriptEntries)
        )
    }
}
