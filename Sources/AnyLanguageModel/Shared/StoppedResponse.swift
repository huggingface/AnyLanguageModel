/// The values that a response which a ``ToolExecutionDelegate`` stopped can end with,
/// so that a response type that can be empty doesn't throw.
///
/// The first value that decodes as the complete content type is used:
/// an empty string for `String`,
/// and otherwise an empty object, `null`, and then an empty array.
/// Zero and `false` aren't in the list,
/// because they would look like the model's answer.
enum StoppedContent {
    static func empty<Content: Generable>(
        _ type: Content.Type
    ) -> (content: Content, rawContent: GeneratedContent)? {
        if type == String.self {
            return ("" as! Content, GeneratedContent(""))
        }
        let candidates: [GeneratedContent.Kind] = [
            .structure(properties: [:], orderedKeys: []), .null, .array([]),
        ]
        for kind in candidates {
            let raw = GeneratedContent(kind: kind)
            if let content = try? Content(raw) { return (content, raw) }
        }
        return nil
    }
}

extension LanguageModel {
    /// The response that ends when a ``ToolExecutionDelegate`` stops a tool call
    /// before the model writes an answer.
    ///
    /// A type that can be empty ends with the first value in ``StoppedContent``.
    /// A type with required properties, or a number or Boolean, can't be empty,
    /// so this throws ``LanguageModelSession/ToolExecutionStopped`` with the entries instead.
    func stoppedResponse<Content: Generable>(
        generating type: Content.Type,
        provider: String,
        transcriptEntries: [Transcript.Entry],
        requests: RequestsUsage,
        providerMetadata: [String: String]? = nil
    ) throws -> LanguageModelSession.Response<Content> {
        guard let empty = StoppedContent.empty(type) else {
            throw LanguageModelSession.ToolExecutionStopped(
                provider: provider,
                transcriptEntries: ArraySlice(transcriptEntries)
            )
        }
        return .init(
            content: empty.content,
            rawContent: empty.rawContent,
            transcriptEntries: ArraySlice(transcriptEntries),
            requests: requests,
            providerMetadata: providerMetadata
        )
    }
}
