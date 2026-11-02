import Testing

@testable import AnyLanguageModel

@Suite("Profile callback serialization")
struct ProfileCallbackSerializationTests {
    // The first request's callback waits until the second request
    // either starts the same callback or waits for the session's callback gate.
    // Without the gate, the second callback starts while the first still runs.
    @Test(arguments: [false, true])
    func overlappingRequestsRunAProfileCallbackOneAtATime(streaming: Bool) async throws {
        let probe = CallbackOverlapProbe()
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Serial") }
                .model(SerialModel())
                .onResponse { await probe.callback() }
        )
        let (events, eventContinuation) = AsyncStream<OverlapEvent>.makeStream()
        await probe.setEventContinuation(eventContinuation)

        let first = Task { try await send("First", to: session, streaming: streaming) }
        await probe.waitUntilTheFirstCallbackStarts()
        let second = Task { try await send("Second", to: session, streaming: streaming) }
        // This task doesn't finish if the second request never waits for the gate.
        Task {
            await session.profileLifecycle.callbackGate.waitUntilContended()
            eventContinuation.yield(.secondRequestWaits)
        }

        var iterator = events.makeAsyncIterator()
        let event = await iterator.next()
        #expect(event == .secondRequestWaits)
        await probe.releaseTheFirstCallback()
        try await first.value
        try await second.value

        let counts = await probe.counts
        #expect(counts == .init(calls: 2, maximumAtOnce: 1))
    }

    // A callback that makes a request on its own session
    // runs that request's callbacks in its own turn instead of waiting for itself.
    @Test(.timeLimit(.minutes(1)), arguments: [false, true])
    func callbackCanMakeARequestOnItsOwnSession(streaming: Bool) async throws {
        let session = Locked<LanguageModelSession?>(nil)
        let responses = Locked<[String]>([])
        @Sendable func followUp(after response: Transcript.Response) async throws {
            let text = response.segments.compactMap { segment -> String? in
                if case .text(let text) = segment { return text.content }
                return nil
            }.joined()
            responses.withLock { $0.append(text) }
            guard responses.withLock({ $0.count }) == 1, let session = session.withLock({ $0 }) else { return }
            try await send("Inner", to: session, streaming: streaming)
        }
        let outer = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Nested") }
                .model(SerialModel())
                .onResponse { response in try await followUp(after: response) }
        )
        session.withLock { $0 = outer }
        defer { session.withLock { $0 = nil } }

        try await send("Outer", to: outer, streaming: streaming)

        #expect(responses.withLock { $0 } == ["response", "response"])
        #expect(outer.transcript.count == 5)
        #expect(outer.properties.history.count == 4)
    }
}

private enum OverlapEvent: Equatable, Sendable {
    case secondCallbackStarts
    case secondRequestWaits
}

/// Records how many profile callbacks run at the same time.
///
/// The first callback waits until the test releases it.
/// Because this is an actor, a second callback can start while the first one waits.
private actor CallbackOverlapProbe {
    struct Counts: Equatable {
        var calls = 0
        var maximumAtOnce = 0
    }

    private(set) var counts = Counts()
    private var running = 0
    private var events: AsyncStream<OverlapEvent>.Continuation?
    private var firstStarted = false
    private var firstStartedWaiters: [CheckedContinuation<Void, Never>] = []
    private var firstRelease: CheckedContinuation<Void, Never>?
    private var isFirstReleased = false

    func setEventContinuation(_ continuation: AsyncStream<OverlapEvent>.Continuation) {
        events = continuation
    }

    func callback() async {
        running += 1
        counts.calls += 1
        counts.maximumAtOnce = max(counts.maximumAtOnce, running)
        if counts.calls == 1 {
            firstStarted = true
            for waiter in firstStartedWaiters { waiter.resume() }
            firstStartedWaiters = []
            if !isFirstReleased {
                await withCheckedContinuation { firstRelease = $0 }
            }
        } else if counts.calls == 2 {
            events?.yield(.secondCallbackStarts)
        }
        running -= 1
    }

    func waitUntilTheFirstCallbackStarts() async {
        guard !firstStarted else { return }
        await withCheckedContinuation { firstStartedWaiters.append($0) }
    }

    func releaseTheFirstCallback() {
        isFirstReleased = true
        firstRelease?.resume()
        firstRelease = nil
    }
}

private func send(_ prompt: String, to session: LanguageModelSession, streaming: Bool) async throws {
    if streaming {
        _ = try await session.streamResponse(to: prompt).collect()
    } else {
        _ = try await session.respond(to: prompt)
    }
}

/// Answers each prompt with "response".
private struct SerialModel: LanguageModel {
    typealias UnavailableReason = Never

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
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
