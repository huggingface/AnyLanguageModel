enum TranscriptEntryKey: Hashable, Sendable {
    case entry(String)
    case toolOutput(round: String?, call: String)
}

// Provider call IDs are scoped to their Tool round, not to the whole transcript.
func keyedTranscriptEntries(_ entries: [Transcript.Entry]) -> [(key: TranscriptEntryKey, entry: Transcript.Entry)] {
    var round: String?
    return entries.map { entry in
        let key: TranscriptEntryKey
        switch entry {
        case .toolCalls(let calls):
            round = calls.id
            key = .entry(entry.id)
        case .toolOutput:
            key = .toolOutput(round: round, call: entry.id)
        case .instructions, .prompt, .response:
            round = nil
            key = .entry(entry.id)
        case .reasoning:
            key = .entry(entry.id)
        }
        return (key, entry)
    }
}

func mergingTranscriptEntries(
    _ original: [Transcript.Entry],
    with additional: [(key: TranscriptEntryKey, entry: Transcript.Entry)]
) -> [Transcript.Entry] {
    var result = original
    var indices: [TranscriptEntryKey: Int] = [:]
    for (index, item) in keyedTranscriptEntries(original).enumerated() { indices[item.key] = index }
    for item in additional {
        if let index = indices[item.key] {
            result[index] = item.entry
        } else {
            indices[item.key] = result.count
            result.append(item.entry)
        }
    }
    return result
}

// Match protected entries by their round-scoped key, not by value,
// so that an identical output from an earlier round stays
// and an edited copy of a pending entry doesn't survive.
// A write can remove a pending output's parent and so change its key;
// an output whose key isn't durable in `original`
// and whose call ID belongs to a protected output is still pending.
// When the new key is a durable output's key, as when an earlier round
// reused the call ID, the durable outputs claim the outputs with that key,
// except for an unchanged pending output.
// A round that isn't in `original` is one that the write added,
// and it owns its outputs, except for an unchanged pending output.
func removingProtectedTranscriptEntries(
    from entries: [Transcript.Entry],
    protecting protectedKeys: Set<TranscriptEntryKey>,
    in original: [Transcript.Entry]
) -> [Transcript.Entry] {
    guard !protectedKeys.isEmpty else { return entries }
    let originalItems = keyedTranscriptEntries(original)
    let durableKeys = Set(originalItems.map(\.key).filter { !protectedKeys.contains($0) })
    let protectedCallIDs = Set(
        originalItems.compactMap { item -> String? in
            guard protectedKeys.contains(item.key), case .toolOutput(_, let call) = item.key else { return nil }
            return call
        }
    )
    let originalRounds = Set(
        originalItems.compactMap { item -> String? in
            guard case .toolCalls(let calls) = item.entry else { return nil }
            return calls.id
        }
    )
    let protectedEntries = originalItems.filter { protectedKeys.contains($0.key) }.map(\.entry)
    let items = keyedTranscriptEntries(entries).filter { !protectedKeys.contains($0.key) }
    var candidates: [Int] = []
    var dropped: Set<Int> = []
    for (index, item) in items.enumerated() {
        guard case .toolOutput(let round, let call) = item.key, protectedCallIDs.contains(call) else { continue }
        if durableKeys.contains(item.key) {
            candidates.append(index)
        } else if let round, !originalRounds.contains(round), !protectedEntries.contains(item.entry) {
            continue
        } else {
            dropped.insert(index)
        }
    }
    let claimed = claimingOutputs(
        in: items,
        at: candidates,
        by: originalItems.filter { durableKeys.contains($0.key) },
        excluding: protectedEntries
    )
    let candidateSet = Set(candidates)
    return items.indices
        .filter { !dropped.contains($0) && (!candidateSet.contains($0) || claimed.contains($0)) }
        .map { items[$0].entry }
}

/// The entries that a response appended to the session history.
struct AppendedTranscriptEntries: Sendable {
    static let none = AppendedTranscriptEntries(keys: [], history: [])

    /// The round-scoped keys of the appended entries.
    let keys: Set<TranscriptEntryKey>

    /// The history right after the append.
    let history: [Transcript.Entry]

    var isEmpty: Bool { keys.isEmpty }
}

// Remove the entries that a response appended from a later version of the history.
// A callback can remove an appended output's parent and so change its key.
// An output with an appended output's call ID that is now in a round
// that the history had before the append, or in no round, is still appended,
// unless an earlier output with that key claims it.
// An unchanged appended output is never claimed.
// A callback can also insert a round before an appended output.
// An unchanged copy of an appended output whose key is gone
// that is now in a round the history didn't have is still appended;
// each such output removes the first copy.
// Other outputs in later rounds belong to later responses and stay.
func removingAppendedTranscriptEntries(
    _ appended: AppendedTranscriptEntries,
    from entries: [Transcript.Entry]
) -> [Transcript.Entry] {
    guard !appended.isEmpty else { return entries }
    let originalItems = keyedTranscriptEntries(appended.history)
    var earlierRounds: Set<String?> = [nil]
    var knownRounds: Set<String> = []
    var appendedCallIDs: Set<String> = []
    for item in originalItems {
        if case .toolCalls(let calls) = item.entry { knownRounds.insert(calls.id) }
        if appended.keys.contains(item.key) {
            if case .toolOutput(_, let call) = item.key { appendedCallIDs.insert(call) }
        } else if case .toolCalls(let calls) = item.entry {
            earlierRounds.insert(calls.id)
        }
    }
    let keyedEntries = keyedTranscriptEntries(entries)
    let remainingKeys = Set(keyedEntries.map(\.key))
    var movedOutputs = originalItems.filter { item in
        guard case .toolOutput = item.key else { return false }
        return appended.keys.contains(item.key) && !remainingKeys.contains(item.key)
    }.map(\.entry)
    let items = keyedEntries.filter { item in
        guard !appended.keys.contains(item.key) else { return false }
        guard case .toolOutput(let round?, _) = item.key, !knownRounds.contains(round),
            let index = movedOutputs.firstIndex(of: item.entry)
        else { return true }
        movedOutputs.remove(at: index)
        return false
    }
    let candidates = items.indices.filter { index in
        guard case .toolOutput(let round, let call) = items[index].key else { return false }
        return appendedCallIDs.contains(call) && earlierRounds.contains(round)
    }
    let claimed = claimingOutputs(
        in: items,
        at: candidates,
        by: originalItems.filter { !appended.keys.contains($0.key) },
        excluding: originalItems.filter { appended.keys.contains($0.key) }.map(\.entry)
    )
    let candidateSet = Set(candidates)
    return items.indices
        .filter { !candidateSet.contains($0) || claimed.contains($0) }
        .map { items[$0].entry }
}

// Each owner claims one candidate with its key:
// one equal to it if there is one, otherwise the first one left
// that isn't equal to an excluded entry.
// An owner that a callback removed can't otherwise tell an edited copy of itself
// from an unchanged pending output that has taken its key,
// so an exact copy of a pending output stays pending.
// Returns the indices of the claimed candidates.
private func claimingOutputs(
    in items: [(key: TranscriptEntryKey, entry: Transcript.Entry)],
    at candidates: [Int],
    by owners: [(key: TranscriptEntryKey, entry: Transcript.Entry)],
    excluding excluded: [Transcript.Entry]
) -> Set<Int> {
    var unclaimed: [TranscriptEntryKey: [Transcript.Entry]] = [:]
    for owner in owners { unclaimed[owner.key, default: []].append(owner.entry) }
    var claimed: Set<Int> = []
    var unmatched: [Int] = []
    for index in candidates {
        let item = items[index]
        if let match = unclaimed[item.key]?.firstIndex(of: item.entry) {
            unclaimed[item.key]?.remove(at: match)
            claimed.insert(index)
        } else {
            unmatched.append(index)
        }
    }
    for index in unmatched
    where unclaimed[items[index].key]?.isEmpty == false && !excluded.contains(items[index].entry) {
        unclaimed[items[index].key]?.removeFirst()
        claimed.insert(index)
    }
    return claimed
}

func toolContinuationMessages<Message>(
    from transcript: Transcript,
    replacing currentRoundEntries: [Transcript.Entry],
    with continuation: [Message],
    hasDynamicProfile: Bool,
    convert: (Transcript) throws -> [Message]
) rethrows -> [Message] {
    guard hasDynamicProfile else { return try convert(transcript) + continuation }
    guard !continuation.isEmpty, !currentRoundEntries.isEmpty else { return try convert(transcript) }
    let projected = Array(transcript)
    let count = currentRoundEntries.count
    let ranges = projected.indices.compactMap { start -> Range<Int>? in
        let end = start + count
        guard end <= projected.count, Array(projected[start ..< end]) == currentRoundEntries else { return nil }
        return start ..< end
    }
    guard ranges.count == 1, let range = ranges.first else {
        // A profile may remove or reorder the current round. Its projection is authoritative.
        return try convert(transcript)
    }
    return try convert(Transcript(entries: projected[..<range.lowerBound]))
        + continuation
        + convert(Transcript(entries: projected[range.upperBound...]))
}

// MARK: - Assistant text in tool rounds

/// Key under which the assistant's text for a tool round is kept in
/// ``Transcript/ToolCalls/providerMetadata``, and the text that a response replays in
/// ``Transcript/Response/providerMetadata``.
///
/// Chat Completions, the OpenAI Responses API, and Open Responses share this key,
/// so a transcript that moves between them replays each piece of text once.
/// The key keeps its original name because saved transcripts contain it.
let assistantContentMetadataKey = "openai.content"

/// Key under which a response keeps its text as written, so that replay can tell
/// when a caller has edited the response since then.
let responseTextMetadataKey = "openai.responseText"

func assistantContentMetadata(_ text: String?) -> [String: String]? {
    guard let text, !text.isEmpty else { return nil }
    return [assistantContentMetadataKey: text]
}

/// The metadata of a string response whose content differs from the text it replays.
///
/// A string response includes the text of every tool round,
/// but replay already sends each round's text with that round's calls.
/// The response keeps only the final round's text,
/// so replay sends each piece of text once, as the requests in flight did.
func responseContentMetadata(_ responseText: String, roundText: String) -> [String: String]? {
    guard responseText != roundText else { return nil }
    return [assistantContentMetadataKey: roundText, responseTextMetadataKey: responseText]
}

/// The metadata of the empty response of a stopped request,
/// which leaves the text of its tool rounds to replay with their calls.
let stoppedResponseContentMetadata = [assistantContentMetadataKey: "", responseTextMetadataKey: ""]

/// The metadata of a response that a model handoff continues after `prefixEntries`.
///
/// The handoff adds `prefixText` to the response's content.
/// Replay sends the part of it that tool rounds carry with their calls.
/// The rest has no other place, so the response keeps it, before its own text.
func handoffResponseMetadata(
    _ metadata: [String: String]?,
    responseText: String,
    prefixText: String,
    prefixEntries: [Transcript.Entry]
) -> [String: String]? {
    guard !prefixText.isEmpty else { return metadata }
    var uncarried = ""
    var remaining = prefixText[...]
    for case .toolCalls(let calls) in prefixEntries {
        guard let text = calls.providerMetadata?[assistantContentMetadataKey],
            let range = remaining.range(of: text)
        else { continue }
        uncarried += remaining[..<range.lowerBound]
        remaining = remaining[range.upperBound...]
    }
    uncarried += remaining
    let combined = prefixText + responseText
    let replayed = uncarried + (metadata?[assistantContentMetadataKey] ?? responseText)
    guard replayed != combined else { return metadata }
    return (metadata ?? [:]).merging(
        [assistantContentMetadataKey: replayed, responseTextMetadataKey: combined]
    ) { _, new in new }
}

extension Transcript {
    /// The entries to send on replay, with each turn's assistant text resolved:
    /// each response has the segments to send,
    /// and only tool rounds whose text is still current have it in their metadata.
    ///
    /// A string response shows the caller the text of its turn's tool rounds,
    /// and a caller can edit, rebuild, or remove the response but can't see the metadata.
    /// So the metadata applies only to a response that is unchanged since it was written.
    /// Otherwise, the response's segments are the turn's text,
    /// and the rounds send only their calls.
    /// The same goes for a finished turn without a response.
    /// A structured response doesn't include its rounds' text, so they keep it.
    var replayEntries: [Entry] {
        var entries = Array(self)
        for turn in assistantTurns(entries) {
            resolveAssistantText(in: &entries, turn: turn.range, isLast: turn.isLast)
        }
        return entries
    }

    private func resolveAssistantText(in entries: inout [Entry], turn: Range<Int>, isLast: Bool) {
        if case .prompt(let prompt) = entries[turn.lowerBound], prompt.responseFormat != nil { return }
        let responses = entries[turn].compactMap { entry -> Response? in
            if case .response(let response) = entry { response } else { nil }
        }
        let isCurrent = responses.isEmpty ? isLast : responses.allSatisfy(\.isUnchanged)
        for index in turn {
            switch entries[index] {
            case .response(var response):
                guard isCurrent, let text = response.providerMetadata?[assistantContentMetadataKey] else { continue }
                response.segments = [.text(.init(content: text))]
                entries[index] = .response(response)
            case .toolCalls(var calls):
                guard !isCurrent else { continue }
                calls.providerMetadata?[assistantContentMetadataKey] = nil
                entries[index] = .toolCalls(calls)
            default:
                continue
            }
        }
    }
}

/// The ranges of the turns in `entries`, each from a prompt up to the next prompt.
func assistantTurns(_ entries: [Transcript.Entry]) -> [(range: Range<Int>, isLast: Bool)] {
    var turns: [(range: Range<Int>, isLast: Bool)] = []
    var start = entries.startIndex
    while start < entries.endIndex {
        let next = entries.indices.dropFirst(start + 1).first {
            if case .prompt = entries[$0] { true } else { false }
        }
        let end = next ?? entries.endIndex
        turns.append((start ..< end, next == nil))
        start = end
    }
    return turns
}

extension Transcript.Response {
    /// Whether the response's segments are the text that its metadata was written for.
    fileprivate var isUnchanged: Bool {
        guard let written = providerMetadata?[responseTextMetadataKey],
            providerMetadata?[assistantContentMetadataKey] != nil
        else { return false }
        var text = ""
        for segment in segments {
            guard case .text(let segment) = segment else { return false }
            text += segment.content
        }
        return text == written
    }
}
