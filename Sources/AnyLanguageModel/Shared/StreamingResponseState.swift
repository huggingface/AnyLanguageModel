/// Accumulates response metadata across streamed tool rounds.
struct StreamingResponseState<Content: Generable> {
    /// The text of the current round, which providers replay as that round's assistant message.
    var text = ""
    var entries: [Transcript.Entry] = []
    var usage = ReportedUsage()
    private var completedUsage = LanguageModelSession.Usage.zero

    /// The text of completed rounds, for string content.
    ///
    /// Like the MLX and llama.cpp providers,
    /// a string response includes the text from every round.
    /// Structured content uses only the current round,
    /// because text from separate rounds doesn't form one JSON value.
    private var earlierText = ""

    var totalUsage: LanguageModelSession.Usage {
        var total = completedUsage
        total.add(usage.value)
        return total
    }

    /// The response text so far.
    var responseText: String {
        Content.self == String.self ? earlierText + text : text
    }

    func snapshot(providerMetadata: [String: String]? = nil) -> LanguageModelSession.ResponseStream<Content>.Snapshot? {
        guard
            var snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(text: responseText, usage: totalUsage)
        else { return nil }
        snapshot.transcriptEntries = ArraySlice(entries)
        snapshot.providerMetadata = providerMetadata
        return snapshot
    }

    /// A stopped tool call can have no response text, including for structured generation.
    ///
    /// Without text, the snapshot uses the first empty value that the content type accepts:
    /// an empty object, an empty array, `null`, and then zero or `false` for scalar types.
    /// The value must decode as the complete content type, not only its partial form,
    /// so that `collect()` accepts the snapshot.
    func stoppedSnapshot() throws -> LanguageModelSession.ResponseStream<Content>.Snapshot {
        if let snapshot = snapshot() { return snapshot }
        let candidates: [GeneratedContent.Kind] = [
            .structure(properties: [:], orderedKeys: []), .array([]), .null, .number(0), .bool(false),
        ]
        for kind in candidates {
            let raw = GeneratedContent(kind: kind)
            guard let content = try? Content(raw) else { continue }
            return .init(
                content: content.asPartiallyGenerated(),
                rawContent: raw,
                transcriptEntries: ArraySlice(entries),
                usage: totalUsage
            )
        }
        throw GeneratedContentError.typeMismatch
    }

    mutating func beginNextRound() {
        completedUsage.add(usage.value)
        usage = ReportedUsage()
        earlierText = responseText
        text = ""
    }
}
