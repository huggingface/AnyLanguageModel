import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

/// Each test pauses profile resolution or a profile callback after it reads the history,
/// changes the history from outside the request,
/// and then lets the profile continue.
/// A paused resolution runs on a ``BlockingTaskExecutor``,
/// so the pause doesn't block a thread of the cooperative pool;
/// a callback is asynchronous and pauses without blocking.
@Suite("Profile history conflicts")
struct ProfileHistoryConflictTests {
    @available(macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
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
        let request = Task.detached(executorPreference: BlockingTaskExecutor()) {
            try await send("A", to: session, streaming: streaming)
        }
        await probe.waitUntilPaused()
        await other.release.open()
        _ = try await otherRequest.value
        probe.resume()

        await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileResolving) {
            try await request.value
        }
        #expect(texts(session.transcript) == ["Seed", "B", "response"])
    }

    @available(macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test(arguments: [false, true])
    func directHistoryWriteSurvivesAProfileEdit(streaming: Bool) async throws {
        let probe = ConflictProbe(pausesAfter: "A", edits: true)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel()),
            history: [seed]
        )

        let request = Task.detached(executorPreference: BlockingTaskExecutor()) {
            try await send("A", to: session, streaming: streaming)
        }
        await probe.waitUntilPaused()
        session.properties.history = [seed, direct]
        probe.resume()

        await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileResolving) {
            try await request.value
        }
        #expect(texts(session.transcript) == ["Seed", "Direct"])
    }

    @available(macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test(arguments: [false, true])
    func profileWithoutAnEditAppendsThePrompt(streaming: Bool) async throws {
        let probe = ConflictProbe(pausesAfter: "A", edits: false)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel()),
            history: [seed]
        )

        let request = Task.detached(executorPreference: BlockingTaskExecutor()) {
            try await send("A", to: session, streaming: streaming)
        }
        await probe.waitUntilPaused()
        session.properties.history = [seed, direct]
        probe.resume()

        try await request.value
        #expect(texts(session.transcript) == ["Seed", "Direct", "A", "response"])
    }

    @available(macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test(arguments: [false, true])
    func resolutionEditDoesNotReplaceAChangedHistory(throwing: Bool) async throws {
        let probe = ConflictProbe(pausesAfter: "Seed", edits: true)
        let session = LanguageModelSession(
            profile: ConflictProfile(probe: probe, model: ConflictModel()),
            history: [seed]
        )

        let resolution = Task.detached(executorPreference: BlockingTaskExecutor()) {
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

    @Test(arguments: CallbackKind.allCases)
    func directHistoryWriteSurvivesACallbackEdit(kind: CallbackKind) async throws {
        let gates = ConflictRequestGates()
        let session = LanguageModelSession(
            profile: CallbackConflictProfile(pausing: kind, gates: gates),
            history: [seed]
        )
        let call = Transcript.ToolCall(id: "call", toolName: "fixture", arguments: .init(properties: [:]))
        let calls = Transcript.Entry.toolCalls(.init([call]))
        let output = Transcript.ToolOutput(id: call.id, toolName: call.toolName, segments: [])
        let reasoning = Transcript.Entry.reasoning(.init(segments: [.text(.init(content: "Thinking"))]))
        let response = Transcript.Entry.response(.init(assetIDs: [], segments: [.text(.init(content: "Answer"))]))

        // Activation and `onPrompt` run while the request context resolves;
        // the other callbacks run with a context that resolved first.
        let context: LanguageModelSession.RequestContext? =
            kind.isLifecycle ? nil : try await session.resolvedRequestContext(including: [])
        let request = Task {
            guard let context else {
                _ = try await session.resolvedRequestContext(including: [])
                return
            }
            switch kind {
            case .toolCall:
                try await session.profileWillExecuteToolCall(
                    call,
                    requestContext: context,
                    currentRoundEntries: [calls]
                )
            case .toolOutput:
                try await session.profileDidProduceToolOutput(
                    for: call,
                    output: output,
                    requestContext: context,
                    currentRoundEntries: [calls, .toolOutput(output)]
                )
            case .reasoning, .response:
                try await session.profileDidProduceEntries(
                    [kind == .reasoning ? reasoning : response],
                    requestContext: context,
                    currentRoundEntries: []
                )
            case .activate, .prompt:
                break
            }
        }
        await gates.entered.wait()
        session.properties.history = [seed, direct]
        await gates.release.open()

        await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileRunningCallback) {
            try await request.value
        }
        #expect(texts(session.transcript) == ["Seed", "Direct"])
    }

    // A stale `onResponse` write must not delete an overlapping request's turn.
    // The overlapping request adds its prompt,
    // then waits for the paused callback before its own profile actions run.
    @Test(arguments: [false, true])
    func concurrentResponseSurvivesAResponseCallbackEdit(streaming: Bool) async throws {
        let gates = ConflictRequestGates()
        let session = LanguageModelSession(
            profile: CallbackConflictProfile(pausing: .response, gates: gates, rewritesSnapshotOnce: true),
            history: [seed]
        )

        let request = Task { try await send("A", to: session, streaming: streaming) }
        await gates.entered.wait()
        let overlapping = Task { try await send("B", to: session, streaming: streaming) }
        await session.profileLifecycle.callbackGate.waitUntilContended()
        await gates.release.open()

        await #expect(throws: LanguageModelSession.DynamicProfileError.historyMutationWhileRunningCallback) {
            try await request.value
        }
        try await overlapping.value
        #expect(texts(session.transcript) == ["Seed", "A", "B", "response"])
    }

    // Two callbacks in one request run one at a time,
    // and the later one reads the history after the earlier one's write,
    // so the later write doesn't undo the earlier one.
    @Test func staleCallbackSnapshotDoesNotUndoAnotherCallbackEdit() async throws {
        let gates = ConflictRequestGates()
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Sibling callbacks") }
                .model(ConflictModel())
                .onToolCall { call in
                    guard let properties = SessionPropertyBinding.values else {
                        Issue.record("Expected session properties in the callback")
                        return
                    }
                    let history = properties.history
                    if call.id == "paused" {
                        await gates.entered.open()
                        await gates.release.wait()
                    }
                    let edit = Transcript.Entry.prompt(.init(segments: [.text(.init(content: call.id))]))
                    properties.history = history + [edit]
                },
            history: [seed]
        )
        let context = try await session.resolvedRequestContext(including: [])
        func call(_ id: String) -> Transcript.ToolCall {
            Transcript.ToolCall(id: id, toolName: "fixture", arguments: .init(properties: [:]))
        }

        let paused = Task {
            try await session.profileWillExecuteToolCall(
                call("paused"),
                requestContext: context,
                currentRoundEntries: []
            )
        }
        await gates.entered.wait()
        let second = Task {
            try await session.profileWillExecuteToolCall(
                call("second"),
                requestContext: context,
                currentRoundEntries: []
            )
        }
        await session.profileLifecycle.callbackGate.waitUntilContended()
        await gates.release.open()
        try await paused.value
        try await second.value

        #expect(texts(session.transcript) == ["Seed", "paused", "second"])
    }

    @Test func conflictErrorsHaveMessages() {
        let resolving = LanguageModelSession.DynamicProfileError.historyMutationWhileResolving
        #expect(resolving.errorDescription?.contains("history changed") == true)
        let callback = LanguageModelSession.DynamicProfileError.historyMutationWhileRunningCallback
        #expect(callback.errorDescription?.contains("callback") == true)
    }
}

enum CallbackKind: CaseIterable, Sendable {
    case activate, prompt, toolCall, toolOutput, reasoning, response

    var isLifecycle: Bool { self == .activate || self == .prompt }
}

/// Pauses one kind of callback until its gate opens,
/// and then removes the seed prompt from the history that the callback saw.
/// With `rewritesSnapshotOnce`, only the first such callback pauses,
/// and it writes back the history that it saw, as `history = history` would.
private struct CallbackConflictProfile: LanguageModelSession.DynamicProfile {
    let pausing: CallbackKind
    let gates: ConflictRequestGates
    var rewritesSnapshotOnce = false
    private let hasEdited = Locked(false)

    init(pausing: CallbackKind, gates: ConflictRequestGates, rewritesSnapshotOnce: Bool = false) {
        self.pausing = pausing
        self.gates = gates
        self.rewritesSnapshotOnce = rewritesSnapshotOnce
    }

    var body: some LanguageModelSession.DynamicProfile {
        let pausing = pausing
        let gates = gates
        let rewritesSnapshotOnce = rewritesSnapshotOnce
        let hasEdited = hasEdited
        @Sendable func pauseAndEdit(_ kind: CallbackKind) async {
            guard kind == pausing else { return }
            if rewritesSnapshotOnce {
                let isFirst = hasEdited.withLock { hasEdited in
                    defer { hasEdited = true }
                    return !hasEdited
                }
                guard isFirst else { return }
            }
            guard let properties = SessionPropertyBinding.values else {
                Issue.record("Expected session properties in the callback")
                return
            }
            let history = properties.history
            await gates.entered.open()
            await gates.release.wait()
            properties.history = rewritesSnapshotOnce ? history : history.filter { text(of: $0) != "Seed" }
        }
        return LanguageModelSession.Profile { Instructions("Callback conflicts") }
            .model(ConflictModel())
            .onActivate { await pauseAndEdit(.activate) }
            .onPrompt { await pauseAndEdit(.prompt) }
            .onToolCall { await pauseAndEdit(.toolCall) }
            .onToolOutput { await pauseAndEdit(.toolOutput) }
            .onReasoning { await pauseAndEdit(.reasoning) }
            .onResponse { await pauseAndEdit(.response) }
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
    private let hasSkippedInitialEvaluation = Locked(false)

    init(pausesAfter: String, edits: Bool) {
        self.pausesAfter = pausesAfter
        self.edits = edits
        (paused, pausedContinuation) = AsyncStream<Void>.makeStream()
    }

    /// Returns `true` if the body paused and should now make its edit.
    ///
    /// The session evaluates the body once when it's created,
    /// to record the profile's instructions;
    /// the probe never pauses that evaluation.
    func pauseIfNeeded(history: [Transcript.Entry]) -> Bool {
        let isInitialEvaluation = hasSkippedInitialEvaluation.withLock { hasSkipped in
            defer { hasSkipped = true }
            return !hasSkipped
        }
        guard !isInitialEvaluation, let last = history.last, text(of: last) == pausesAfter else { return false }
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
