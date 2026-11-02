import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

/// Each test pauses profile resolution after it reads the history,
/// changes the history from outside the resolution,
/// and then lets the profile continue.
@Suite("Profile history conflicts")
struct ProfileHistoryConflictTests {
    @Test(arguments: [false, true])
    func concurrentResponseSurvivesAProfileEdit(streaming: Bool) async throws {
        let other = ConflictRequestGates()
        let probe = ConflictProbe(pausesAfter: "A", edits: true)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel(gatedPrompt: "B", gates: other)),
            history: [seed]
        )

        let otherRequest = Task { try await session.respond(to: "B") }
        await other.entered.wait()
        let request = Task.detached { try await send("A", to: session, streaming: streaming) }
        await probe.waitUntilPaused()
        await other.release.open()
        _ = try await otherRequest.value
        probe.resume()

        await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileResolving) {
            try await request.value
        }
        #expect(texts(session.transcript) == ["Seed", "B", "response"])
    }

    @Test(arguments: [false, true])
    func directHistoryWriteSurvivesAProfileEdit(streaming: Bool) async throws {
        let probe = ConflictProbe(pausesAfter: "A", edits: true)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel()),
            history: [seed]
        )

        let request = Task.detached { try await send("A", to: session, streaming: streaming) }
        await probe.waitUntilPaused()
        session.properties.history = [seed, direct]
        probe.resume()

        await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileResolving) {
            try await request.value
        }
        #expect(texts(session.transcript) == ["Seed", "Direct"])
    }

    @Test(arguments: [false, true])
    func profileWithoutAnEditAppendsThePrompt(streaming: Bool) async throws {
        let probe = ConflictProbe(pausesAfter: "A", edits: false)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel()),
            history: [seed]
        )

        let request = Task.detached { try await send("A", to: session, streaming: streaming) }
        await probe.waitUntilPaused()
        session.properties.history = [seed, direct]
        probe.resume()

        try await request.value
        #expect(texts(session.transcript) == ["Seed", "Direct", "A", "response"])
    }

    @Test(arguments: [false, true])
    func resolutionEditDoesNotReplaceAChangedHistory(throwing: Bool) async throws {
        let probe = ConflictProbe(pausesAfter: "Seed", edits: true)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel()),
            history: [seed]
        )

        let resolution = Task.detached {
            if throwing {
                _ = try await session.resolvedRequestContext(including: [])
            } else {
                // The non-throwing overload keeps the edits in the returned context only.
                let context = session.resolvedRequestContext()
                #expect(texts(context.transcript).isEmpty)
            }
        }
        await probe.waitUntilPaused()
        session.properties.history = [seed, direct]
        probe.resume()

        if throwing {
            await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileResolving) {
                try await resolution.value
            }
        } else {
            try await resolution.value
        }
        #expect(texts(session.transcript) == ["Seed", "Direct"])
    }

    @Test func conflictErrorHasAMessage() {
        let error = LanguageModelSession.DynamicProfileError.historyMutationWhileResolving
        #expect(error.errorDescription?.contains("history changed") == true)
    }
}

private let seed = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Seed"))]))
private let direct = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Direct"))]))

private func send(_ prompt: String, to session: LanguageModelSession, streaming: Bool) async throws {
    if streaming {
        _ = try await session.streamResponse(to: prompt).collect()
    } else {
        _ = try await session.respond(to: prompt)
    }
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

private func texts(_ transcript: Transcript) -> [String] {
    transcript.compactMap(text(of:))
}

/// Pauses the profile body once, when the last entry in its history is a given prompt,
/// and optionally removes the seed prompt after the pause.
private final class ConflictProbe: Sendable {
    let pausesAfter: String
    let edits: Bool
    private let paused: AsyncStream<Void>
    private let pausedContinuation: AsyncStream<Void>.Continuation
    private let semaphore = DispatchSemaphore(value: 0)
    private let hasPaused = Locked(false)

    init(pausesAfter: String, edits: Bool) {
        self.pausesAfter = pausesAfter
        self.edits = edits
        (paused, pausedContinuation) = AsyncStream<Void>.makeStream()
    }

    /// Returns `true` if the body paused and should now make its edit.
    func pauseIfNeeded(history: [Transcript.Entry]) -> Bool {
        guard let last = history.last, text(of: last) == pausesAfter else { return false }
        let shouldPause = hasPaused.withLock { hasPaused in
            defer { hasPaused = true }
            return !hasPaused
        }
        guard shouldPause else { return false }
        pausedContinuation.yield()
        semaphore.wait()
        return edits
    }

    func waitUntilPaused() async {
        for await _ in paused.prefix(1) {}
    }

    func resume() {
        semaphore.signal()
    }
}

private struct ConflictProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let probe: ConflictProbe
    let model: ConflictModel
    @SessionProperty(\.history) private var history

    var body: some LanguageModelSession.DynamicProfile {
        if probe.pauseIfNeeded(history: Array(history)) {
            history.removeAll { text(of: $0) == "Seed" }
        }
        return LanguageModelSession.Profile { Instructions("Conflicts") }
            .model(model)
    }
}

private actor ConflictGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

private struct ConflictRequestGates: Sendable {
    let entered = ConflictGate()
    let release = ConflictGate()
}

/// Answers each prompt with "response",
/// waiting for its gates first when the prompt is the gated one.
private struct ConflictModel: LanguageModel {
    typealias UnavailableReason = Never
    var gatedPrompt: String?
    var gates = ConflictRequestGates()

    private func waitIfGated(_ prompt: Prompt) async {
        guard prompt.description == gatedPrompt else { return }
        await gates.entered.open()
        await gates.release.wait()
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        await waitIfGated(prompt)
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
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            let task = Task {
                await waitIfGated(prompt)
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
            continuation.onTermination = { _ in task.cancel() }
        }
        return .init(stream: stream)
    }
}
