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

func removingProtectedTranscriptEntries(
    from entries: [Transcript.Entry],
    protecting protected: [Transcript.Entry]
) -> [Transcript.Entry] {
    var result = entries
    for entry in protected.reversed() {
        if let index = result.lastIndex(of: entry) { result.remove(at: index) }
    }
    return result
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
