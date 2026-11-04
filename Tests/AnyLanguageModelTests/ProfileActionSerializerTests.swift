import Testing

@testable import AnyLanguageModel

@Suite("Profile action serializer")
struct ProfileActionSerializerTests {
    // The first request's callback waits until the second request
    // either starts the same callback or waits for the session's permit.
    // Without the permit, the second callback starts while the first still runs.
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
        // This task doesn't finish if the second request never waits for the permit.
        Task {
            await session.profileLifecycle.actionSerializer.waitUntilContended()
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
    // releases the session's permit while it waits,
    // so that request's callbacks run instead of waiting for the callback.
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

    enum FollowUp: CaseIterable, Sendable {
        case respondInATask, streamInATask, unconsumedStream
    }

    // A callback that starts a request on its own session without awaiting it
    // keeps running while that request runs.
    // The first callback waits until the follow-up request
    // either starts the same callback or waits for the permit.
    // Without the permit, the follow-up callback starts while the first still runs.
    @Test(.timeLimit(.minutes(1)), arguments: FollowUp.allCases, [false, true])
    func followUpRequestWaitsForTheCallbackThatStartsIt(followUp: FollowUp, streaming: Bool) async throws {
        let probe = CallbackOverlapProbe()
        let session = Locked<LanguageModelSession?>(nil)
        let followUpTask = Locked<Task<Void, any Error>?>(nil)
        let pendingStream = Locked<LanguageModelSession.ResponseStream<String>?>(nil)
        @Sendable func startFollowUp() {
            guard let session = session.withLock({ $0 }) else { return }
            switch followUp {
            case .respondInATask:
                followUpTask.withLock { $0 = Task { try await send("Follow-up", to: session, streaming: false) } }
            case .streamInATask:
                followUpTask.withLock { $0 = Task { try await send("Follow-up", to: session, streaming: true) } }
            case .unconsumedStream:
                pendingStream.withLock { $0 = session.streamResponse(to: "Follow-up") }
            }
        }
        let outer = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Follow-up") }
                .model(SerialModel())
                .onResponse { await probe.callback(onFirstStart: startFollowUp) }
        )
        session.withLock { $0 = outer }
        defer { session.withLock { $0 = nil } }
        let (events, eventContinuation) = AsyncStream<OverlapEvent>.makeStream()
        await probe.setEventContinuation(eventContinuation)

        let first = Task { try await send("First", to: outer, streaming: streaming) }
        await probe.waitUntilTheFirstCallbackStarts()
        // This task doesn't finish if the follow-up request never waits for the permit.
        Task {
            await outer.profileLifecycle.actionSerializer.waitUntilContended()
            eventContinuation.yield(.secondRequestWaits)
        }

        var iterator = events.makeAsyncIterator()
        let event = await iterator.next()
        #expect(event == .secondRequestWaits)
        await probe.releaseTheFirstCallback()
        try await first.value
        try await followUpTask.withLock({ $0 })?.value
        if let stream = pendingStream.withLock({ $0 }) {
            _ = try await stream.collect()
        }

        let counts = await probe.counts
        #expect(counts == .init(calls: 2, maximumAtOnce: 1))
        #expect(outer.properties.history.count == 4)
    }

    // A callback whose request on its own session makes another request
    // gets both responses, whether each request streams or not.
    @Test(.timeLimit(.minutes(1)), arguments: [false, true], [false, true])
    func nestedCallbacksCanEachMakeARequest(outerStreaming: Bool, innerStreaming: Bool) async throws {
        let session = Locked<LanguageModelSession?>(nil)
        let calls = Locked(0)
        let running = Locked((current: 0, maximum: 0))
        @Sendable func followUp() async throws {
            running.withLock {
                $0.current += 1
                $0.maximum = max($0.maximum, $0.current)
            }
            defer { running.withLock { $0.current -= 1 } }
            let call = calls.withLock { calls in
                calls += 1
                return calls
            }
            guard call < 3, let session = session.withLock({ $0 }) else { return }
            try await send("Nested \(call)", to: session, streaming: call == 1 ? innerStreaming : outerStreaming)
        }
        let outer = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Nested") }
                .model(SerialModel())
                .onResponse { _ in try await followUp() }
        )
        session.withLock { $0 = outer }
        defer { session.withLock { $0 = nil } }

        try await send("Outer", to: outer, streaming: outerStreaming)

        #expect(calls.withLock { $0 } == 3)
        // Each callback awaits the request that it makes, so the callbacks nest.
        #expect(running.withLock { $0.maximum } == 3)
        #expect(outer.properties.history.count == 6)
    }
}

extension ProfileActionSerializerTests {
    // While a callback waits for a request on another session,
    // a callback of an unrelated request on its session runs.
    // The first callback continues only after the second one ends,
    // so their code never runs at the same time.
    @Test(.timeLimit(.minutes(1)), arguments: [false, true])
    func unrelatedCallbackRunsWhileACallbackWaitsForARequest(streaming: Bool) async throws {
        let events = Locked<[String]>([])
        let (other, otherModel) = (Locked<LanguageModelSession?>(nil), HeldModel())
        let (secondStarted, secondStartedContinuation) = AsyncStream<Void>.makeStream()
        let (secondRelease, secondReleaseContinuation) = AsyncStream<Void>.makeStream()
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Reentrant") }
                .model(SerialModel())
                .onResponse { _ in
                    let isFirst = events.withLock { events in
                        defer { events.append("start") }
                        return !events.contains("start")
                    }
                    if isFirst {
                        guard let other = other.withLock({ $0 }) else { return }
                        events.withLock { $0.append("first waits") }
                        try await send("Nested", to: other, streaming: streaming)
                        events.withLock { $0.append("first continues") }
                    } else {
                        secondStartedContinuation.yield()
                        for await _ in secondRelease { break }
                        events.withLock { $0.append("second ends") }
                    }
                }
        )
        other.withLock { $0 = LanguageModelSession(model: otherModel) }
        defer { other.withLock { $0 = nil } }

        let outcome = await outcome(within: 30) {
            let first = Task { try await send("First", to: session, streaming: streaming) }
            await otherModel.waitUntilARequestStarts()
            let second = Task { try await send("Second", to: session, streaming: streaming) }
            for await _ in secondStarted { break }
            // The first callback's request ends, but the second callback still runs.
            await otherModel.release()
            await session.profileLifecycle.actionSerializer.waitUntilContended()
            secondReleaseContinuation.yield()
            try await first.value
            try await second.value
        }

        try #require(outcome, "The requests didn't finish").get()

        #expect(events.withLock { $0 } == ["start", "first waits", "start", "second ends", "first continues"])
    }

    // A cancelled task that waits for the permit throws `CancellationError`
    // without running its operation, whether it inherits the holder or not.
    @Test(.timeLimit(.minutes(1)), arguments: [false, true])
    func queuedCancellationPreservesExclusion(inherited: Bool) async throws {
        let serializer = ProfileActionSerializer()
        let ran = Locked(false)
        let outcome = await outcome(within: 30) {
            try await serializer.run {
                let waiter: Task<Void, any Error>
                if inherited {
                    waiter = Task { try await serializer.run { ran.withLock { $0 = true } } }
                } else {
                    waiter = Task.detached { try await serializer.run { ran.withLock { $0 = true } } }
                }
                await serializer.waitUntilContended()
                waiter.cancel()
                switch await waiter.result {
                case .success: Issue.record("The cancelled waiter ran")
                case .failure(let error): #expect(error is CancellationError)
                }
                #expect(!ran.withLock { $0 })
            }
            try await serializer.run { ran.withLock { $0 = true } }
        }

        try #require(outcome, "The operations didn't finish").get()
        #expect(ran.withLock { $0 })
    }

    // A callback that consumes part of a stream and drops it
    // holds the permit again when it continues,
    // so a nested operation in its task runs without waiting.
    @Test(.timeLimit(.minutes(1)))
    func partialIterationTakesBackThePermitBeforeReturning() async throws {
        let session = Locked<LanguageModelSession?>(nil)
        let calls = Locked(0)
        let outer = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Partial stream") }
                .model(SerialModel())
                .onResponse { _ in
                    let isFirst = calls.withLock { calls in
                        calls += 1
                        return calls == 1
                    }
                    guard isFirst, let session = session.withLock({ $0 }) else { return }
                    do {
                        let stream = session.streamResponse(to: "Partly consumed")
                        var iterator = stream.makeAsyncIterator()
                        #expect(try await iterator.next() != nil)
                    }
                    try await session.profileLifecycle.actionSerializer.run {}
                }
        )
        session.withLock { $0 = outer }
        defer { session.withLock { $0 = nil } }

        let outcome = await outcome(within: 30) {
            _ = try await outer.respond(to: "Outer")
            _ = try await outer.respond(to: "After partial stream")
        }

        try #require(outcome, "The requests didn't finish").get()
    }

    enum Handoff: CaseIterable, Sendable {
        case release, releaseAndTakeBack
    }

    // Cancelling a waiter while the permit passes to it
    // never lets two operations run at the same time.
    @Test(.timeLimit(.minutes(1)), arguments: Handoff.allCases)
    func cancellationRacesWithHandoff(handoff: Handoff) async throws {
        let outcome = await outcome(within: 30) {
            try await Self.raceCancellationWithHandoff(handoff, iterations: 100)
        }

        try #require(outcome, "The operations didn't finish").get()
    }

    private static func raceCancellationWithHandoff(_ handoff: Handoff, iterations: Int) async throws {
        for _ in 0 ..< iterations {
            let serializer = ProfileActionSerializer()
            let active = Locked(false)
            let tasks = try await serializer.run {
                active.withLock { $0 = true }
                @Sendable func operation() async throws {
                    try await serializer.run {
                        active.withLock {
                            #expect(!$0)
                            $0 = true
                        }
                        await Task.yield()
                        active.withLock { $0 = false }
                    }
                }
                let waiter = Task.detached { try await operation() }
                await serializer.waitUntilContended()
                let cancellation = Task.detached {
                    await Task.yield()
                    waiter.cancel()
                }
                active.withLock { $0 = false }
                if handoff == .releaseAndTakeBack {
                    try await ProfileActionSerializer.withPermitsReleased { await Task.yield() }
                    active.withLock {
                        #expect(!$0)
                        $0 = true
                    }
                    active.withLock { $0 = false }
                }
                return (waiter, cancellation)
            }
            _ = await tasks.0.result
            await tasks.1.value
            try await serializer.run { #expect(!active.withLock { $0 }) }
        }
    }
}

/// Records when a ``HeldModel`` request starts, and holds it until the test releases it.
private actor HeldModelState {
    private var isReleased = false
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []
    private var hasStarted = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []

    func start() async {
        hasStarted = true
        for waiter in startWaiters { waiter.resume() }
        startWaiters = []
        guard !isReleased else { return }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }

    func waitUntilARequestStarts() async {
        guard !hasStarted else { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }

    func release() {
        isReleased = true
        for waiter in releaseWaiters { waiter.resume() }
        releaseWaiters = []
    }
}

/// Answers each prompt with "response" after the test releases it.
private struct HeldModel: LanguageModel {
    typealias UnavailableReason = Never

    let state = HeldModelState()

    func waitUntilARequestStarts() async { await state.waitUntilARequestStarts() }
    func release() async { await state.release() }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        await state.start()
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
        let state = state
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            let task = Task {
                await state.start()
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

    func callback(onFirstStart: @Sendable () -> Void = {}) async {
        running += 1
        counts.calls += 1
        counts.maximumAtOnce = max(counts.maximumAtOnce, running)
        if counts.calls == 1 {
            onFirstStart()
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
