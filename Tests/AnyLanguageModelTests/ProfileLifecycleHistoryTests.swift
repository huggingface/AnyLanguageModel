import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

@Suite("Profile lifecycle history")
struct ProfileLifecycleHistoryTests {
    @Test(arguments: [false, true])
    func lifecycleHistoryEditsReachTheCurrentRequest(streaming: Bool) async throws {
        let requests = Locked<[[String]]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Lifecycle edits") }
                .model(RequestRecordingModel(requests: requests))
                .historyTransform { entries in entries.filter { text(of: $0) != "Hidden" } }
                .onActivate {
                    SessionPropertyBinding.values?.history.removeAll { text(of: $0) == "Activated" }
                }
                .onPrompt { _ in
                    SessionPropertyBinding.values?.history.removeAll { text(of: $0) == "Prompted" }
                },
            history: [entry("Kept"), entry("Hidden"), entry("Activated"), entry("Prompted")]
        )

        if streaming {
            _ = try await session.streamResponse(to: "Request").collect()
        } else {
            _ = try await session.respond(to: "Request")
        }

        #expect(requests.withLock { $0 } == [["Kept", "Request"]])
        #expect(session.transcript.compactMap(text(of:)) == ["Kept", "Hidden", "Request", "response"])
    }
}

private func entry(_ content: String) -> Transcript.Entry {
    .prompt(.init(segments: [.text(.init(content: content))]))
}

private func text(of entry: Transcript.Entry) -> String? {
    let segments: [Transcript.Segment]
    switch entry {
    case .prompt(let prompt): segments = prompt.segments
    case .response(let response): segments = response.segments
    default: return nil
    }
    return segments.compactMap { segment in
        if case .text(let text) = segment { return text.content }
        return nil
    }.joined()
}

/// Records the prompt and response texts of each request's transcript,
/// and answers with "response".
private struct RequestRecordingModel: LanguageModel {
    typealias UnavailableReason = Never
    let requests: Locked<[[String]]>

    private func record(_ session: LanguageModelSession) {
        let texts = session.resolvedRequestContext().transcript.compactMap(text(of:))
        requests.withLock { $0.append(texts) }
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        record(session)
        let raw = GeneratedContent("response")
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [])
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        record(session)
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            do {
                let raw = GeneratedContent("response")
                continuation.yield(
                    .init(content: try Content(raw).asPartiallyGenerated(), rawContent: raw, transcriptEntries: [])
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return .init(stream: stream)
    }
}
