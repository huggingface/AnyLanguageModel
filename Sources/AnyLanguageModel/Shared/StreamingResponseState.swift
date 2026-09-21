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
    func stoppedSnapshot() throws -> LanguageModelSession.ResponseStream<Content>.Snapshot {
        if let snapshot = snapshot() { return snapshot }
        let raw = GeneratedContent(properties: [:])
        return .init(
            content: try Content.PartiallyGenerated(raw),
            rawContent: raw,
            transcriptEntries: ArraySlice(entries),
            usage: totalUsage
        )
    }

    mutating func beginNextRound() {
        completedUsage.add(usage.value)
        usage = ReportedUsage()
        text = ""
    }
}
