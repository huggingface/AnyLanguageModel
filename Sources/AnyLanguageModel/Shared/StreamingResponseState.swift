/// Accumulates response metadata across streamed tool rounds.
struct StreamingResponseState<Content: Generable> {
    /// The text of the current round, which providers replay as that round's assistant message.
    var text = ""
    var entries: [Transcript.Entry] = []
    /// The current round's usage.
    var usage = ReportedUsage()
    /// The usage of the rounds before the current one.
    private var earlierRequests = RequestsUsage()

    /// The property that holds the response, if the provider's response schema wraps it.
    let wrapperProperty: String?

    init(wrapperProperty: String? = nil) {
        self.wrapperProperty = wrapperProperty
    }

    /// The text of completed rounds, for string content.
    ///
    /// Like the MLX and llama.cpp providers,
    /// a string response includes the text from every round.
    /// Structured content uses only the current round,
    /// because text from separate rounds doesn't form one JSON value.
    private var earlierText = ""

    /// The usage of each round's request so far, the current one's last.
    var requests: RequestsUsage {
        var requests = earlierRequests
        requests.add(usage.value)
        return requests
    }

    /// The response text so far.
    var responseText: String {
        Content.self == String.self ? earlierText + text : text
    }

    func snapshot(providerMetadata: [String: String]? = nil) -> LanguageModelSession.ResponseStream<Content>.Snapshot? {
        guard
            var snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                text: responseText,
                requests: requests,
                wrapperProperty: wrapperProperty
            )
        else { return nil }
        snapshot.transcriptEntries = ArraySlice(entries)
        snapshot.providerMetadata = providerMetadata
        return snapshot
    }

    /// A stopped tool call can have no response text, including for structured generation.
    ///
    /// When the text so far doesn't decode as the complete content type,
    /// the snapshot uses the first value in ``StoppedContent`` that the content type accepts.
    /// The value must decode as the complete content type, not only its partial form,
    /// so that `collect()` accepts the snapshot.
    /// A type that accepts none of these values
    /// throws ``LanguageModelSession/ToolExecutionStopped`` with the entries instead.
    func stoppedSnapshot(provider: String) throws -> LanguageModelSession.ResponseStream<Content>.Snapshot {
        if let snapshot = snapshot(), (try? Content(snapshot.rawContent)) != nil { return snapshot }
        guard let empty = StoppedContent.empty(Content.self) else {
            throw LanguageModelSession.ToolExecutionStopped(
                provider: provider,
                transcriptEntries: ArraySlice(entries)
            )
        }
        return .init(
            content: empty.content.asPartiallyGenerated(),
            rawContent: empty.rawContent,
            transcriptEntries: ArraySlice(entries),
            requests: requests
        )
    }

    mutating func beginNextRound() {
        earlierRequests.add(usage.value)
        usage = ReportedUsage()
        earlierText = responseText
        text = ""
    }
}
