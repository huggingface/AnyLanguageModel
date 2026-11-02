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
// reused the call ID, each durable output keeps only one output:
// one equal to it if there is one, otherwise the first one left.
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
    var unclaimed: [TranscriptEntryKey: [Transcript.Entry]] = [:]
    for item in originalItems where durableKeys.contains(item.key) {
        if case .toolOutput(_, let call) = item.key, protectedCallIDs.contains(call) {
            unclaimed[item.key, default: []].append(item.entry)
        }
    }

    let items = keyedTranscriptEntries(entries).filter { !protectedKeys.contains($0.key) }
    var keeps = Array(repeating: true, count: items.count)
    var unmatched: [Int] = []
    for (index, item) in items.enumerated() {
        guard case .toolOutput(_, let call) = item.key, protectedCallIDs.contains(call) else { continue }
        guard durableKeys.contains(item.key) else {
            keeps[index] = false
            continue
        }
        if let match = unclaimed[item.key]?.firstIndex(of: item.entry) {
            unclaimed[item.key]?.remove(at: match)
        } else {
            unmatched.append(index)
        }
    }
    for index in unmatched {
        let key = items[index].key
        if unclaimed[key]?.isEmpty == false {
            unclaimed[key]?.removeFirst()
        } else {
            keeps[index] = false
        }
    }
    return zip(items, keeps).filter(\.1).map(\.0.entry)
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
