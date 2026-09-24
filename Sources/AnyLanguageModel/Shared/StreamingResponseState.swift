/// Accumulates response metadata across streamed tool rounds.
struct StreamingResponseState<Content: Generable> {
    var text = ""
    var entries: [Transcript.Entry] = []
    var usage = ReportedUsage()
    private var completedUsage = LanguageModelSession.Usage.zero

    var totalUsage: LanguageModelSession.Usage {
        var total = completedUsage
        total.add(usage.value)
        return total
    }

    func snapshot(providerMetadata: [String: String]? = nil) -> LanguageModelSession.ResponseStream<Content>.Snapshot? {
        guard var snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(text: text, usage: totalUsage)
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
        text = ""
    }
}
