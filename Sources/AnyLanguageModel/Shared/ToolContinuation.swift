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
// reused the call ID, the durable outputs claim the outputs with that key.
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
    let items = keyedTranscriptEntries(entries).filter { !protectedKeys.contains($0.key) }
    var candidates: [Int] = []
    var dropped: Set<Int> = []
    for (index, item) in items.enumerated() {
        guard case .toolOutput(_, let call) = item.key, protectedCallIDs.contains(call) else { continue }
        if durableKeys.contains(item.key) {
            candidates.append(index)
        } else {
            dropped.insert(index)
        }
    }
    let claimed = claimingOutputs(
        in: items,
        at: candidates,
        by: originalItems.filter { durableKeys.contains($0.key) }
    )
    return items.indices
        .filter { !dropped.contains($0) && (!candidates.contains($0) || claimed.contains($0)) }
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
// Outputs in later rounds belong to later responses and stay.
func removingAppendedTranscriptEntries(
    _ appended: AppendedTranscriptEntries,
    from entries: [Transcript.Entry]
) -> [Transcript.Entry] {
    guard !appended.isEmpty else { return entries }
    let originalItems = keyedTranscriptEntries(appended.history)
    var earlierRounds: Set<String?> = [nil]
    var appendedCallIDs: Set<String> = []
    for item in originalItems {
        if appended.keys.contains(item.key) {
            if case .toolOutput(_, let call) = item.key { appendedCallIDs.insert(call) }
        } else if case .toolCalls(let calls) = item.entry {
            earlierRounds.insert(calls.id)
        }
    }
    let items = keyedTranscriptEntries(entries).filter { !appended.keys.contains($0.key) }
    let candidates = items.indices.filter { index in
        guard case .toolOutput(let round, let call) = items[index].key else { return false }
        return appendedCallIDs.contains(call) && earlierRounds.contains(round)
    }
    let claimed = claimingOutputs(
        in: items,
        at: candidates,
        by: originalItems.filter { !appended.keys.contains($0.key) }
    )
    return items.indices
        .filter { !candidates.contains($0) || claimed.contains($0) }
        .map { items[$0].entry }
}

// Each owner claims one candidate with its key:
// one equal to it if there is one, otherwise the first one left.
// Returns the indices of the claimed candidates.
private func claimingOutputs(
    in items: [(key: TranscriptEntryKey, entry: Transcript.Entry)],
    at candidates: [Int],
    by owners: [(key: TranscriptEntryKey, entry: Transcript.Entry)]
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
    for index in unmatched where unclaimed[items[index].key]?.isEmpty == false {
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
