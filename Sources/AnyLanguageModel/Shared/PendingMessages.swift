import Foundation

/// After a round of tool calls, the messages the session's delegate wants added before the
/// next request (``ToolExecutionDelegate/pendingMessages(in:)``), as prompt entries.
///
/// Each entry is built once, here, and a provider sends it in the request with the same
/// conversion it replays a transcript's prompts with, so the request is the same now and when
/// a later response replays it, and a provider's prompt cache still matches.
func pendingPrompts(in session: LanguageModelSession) async throws -> [Transcript.Prompt] {
    guard let delegate = session.toolExecutionDelegate else { return [] }
    return try await delegate.pendingMessages(in: session).map { message in
        let id = UUID().uuidString
        return Transcript.Prompt(id: id, segments: try message.makeTranscriptSegments(occurrenceID: id))
    }
}
