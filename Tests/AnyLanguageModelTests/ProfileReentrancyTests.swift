import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

@Suite("Profile reentrancy")
struct ProfileReentrancyTests {
    enum Hook: CaseIterable, Sendable {
        case activate, prompt
    }

    // A lifecycle action that awaits a request on its own session
    // must get an error instead of waiting for its own preparation.
    @Test(arguments: Hook.allCases, [false, true])
    func lifecycleActionRequestOnTheSameSessionFails(hook: Hook, streaming: Bool) async throws {
        let session = Locked<LanguageModelSession?>(nil)
        let nestedErrors = Locked<[any Error]>([])
        @Sendable func requestFromHook(_ kind: Hook) async {
            guard kind == hook, let session = session.withLock({ $0 }) else { return }
            do {
                try await send("Inner", to: session, streaming: streaming)
                Issue.record("Expected the nested request to fail")
            } catch {
                nestedErrors.withLock { $0.append(error) }
            }
        }
        let outer = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Reentrancy") }
                .model(ReentrancyModel())
                .onActivate { await requestFromHook(.activate) }
                .onPrompt { prompt in
                    guard text(of: prompt) == "Outer" else { return }
                    await requestFromHook(.prompt)
                }
        )
        session.withLock { $0 = outer }
        defer { session.withLock { $0 = nil } }

        let outcome = await outcome(within: 30) {
            try await send("Outer", to: outer, streaming: streaming)
        }

        let result = try #require(outcome, "The outer request didn't finish")
        try result.get()
        let errors = nestedErrors.withLock { $0 }
        #expect(errors.count == 1)
        #expect(
            errors.first as? LanguageModelSession.DynamicProfileError
                == .requestDuringLifecycleAction
        )
    }

    // Session A's lifecycle action awaits a request on session B,
    // whose lifecycle action awaits a request on session A.
    // That request would wait for A's preparation, which waits for B,
    // so it must get an error instead.
    @Test(arguments: Hook.allCases, [false, true])
    func lifecycleActionCycleAcrossSessionsFails(hook: Hook, streaming: Bool) async throws {
        let sessions = Locked<[String: LanguageModelSession]>([:])
        let nestedErrors = Locked<[any Error]>([])
        @Sendable func request(_ prompt: String, on name: String) async {
            guard let session = sessions.withLock({ $0[name] }) else { return }
            do {
                try await send(prompt, to: session, streaming: streaming)
                if prompt == "Inner" { Issue.record("Expected the request on A to fail") }
            } catch {
                nestedErrors.withLock { $0.append(error) }
            }
        }
        func makeSession(_ name: String, trigger: String, then next: (prompt: String, session: String))
            -> LanguageModelSession
        {
            let fired = Locked(false)
            @Sendable func fire(_ kind: Hook, prompt: String) async {
                guard kind == hook, prompt == trigger else { return }
                let isFirst = fired.withLock { fired in
                    defer { fired = true }
                    return !fired
                }
                guard isFirst else { return }
                await request(next.prompt, on: next.session)
            }
            return LanguageModelSession(
                profile: LanguageModelSession.Profile { Instructions(name) }
                    .model(ReentrancyModel())
                    .onActivate { await fire(.activate, prompt: trigger) }
                    .onPrompt { prompt in await fire(.prompt, prompt: text(of: prompt)) }
            )
        }
        let a = makeSession("A", trigger: "Outer", then: ("Middle", "B"))
        let b = makeSession("B", trigger: "Middle", then: ("Inner", "A"))
        sessions.withLock { $0 = ["A": a, "B": b] }
        defer { sessions.withLock { $0 = [:] } }

        let outcome = await outcome(within: 30) {
            try await send("Outer", to: a, streaming: streaming)
        }

        let result = try #require(outcome, "The request on A didn't finish")
        try result.get()
        let errors = nestedErrors.withLock { $0 }
        #expect(errors.count == 1)
        #expect(
            errors.first as? LanguageModelSession.DynamicProfileError
                == .requestDuringLifecycleAction
        )
        #expect(b.properties.history.count == 2)
    }

    // Session A's callback awaits a request on session B,
    // whose callback awaits a request on session A.
    // Each callback waits for the request that it makes, so both finish.
    @Test(arguments: [false, true], [false, true])
    func callbacksOnTwoSessionsCanMakeRequestsOnEachOther(outerStreaming: Bool, innerStreaming: Bool) async throws {
        let sessions = Locked<[String: LanguageModelSession]>([:])
        let calls = Locked<[String]>([])
        func makeSession(_ name: String, then next: String) -> LanguageModelSession {
            LanguageModelSession(
                profile: LanguageModelSession.Profile { Instructions(name) }
                    .model(ReentrancyModel())
                    .onResponse { _ in
                        let isFirst = calls.withLock { calls in
                            defer { calls.append(name) }
                            return !calls.contains(name)
                        }
                        guard isFirst, let session = sessions.withLock({ $0[next] }) else { return }
                        try await send(
                            "From \(name)",
                            to: session,
                            streaming: name == "A" ? innerStreaming : outerStreaming
                        )
                    }
            )
        }
        let a = makeSession("A", then: "B")
        let b = makeSession("B", then: "A")
        sessions.withLock { $0 = ["A": a, "B": b] }
        defer { sessions.withLock { $0 = [:] } }

        let outcome = await outcome(within: 30) {
            try await send("Outer", to: a, streaming: outerStreaming)
        }

        let result = try #require(outcome, "The request on A didn't finish")
        try result.get()
        #expect(calls.withLock { $0 } == ["A", "B", "A"])
        #expect(a.properties.history.count == 4)
        #expect(b.properties.history.count == 2)
    }

    // Session A's callback awaits a request on session B,
    // while B's callback, from an unrelated request, awaits a request on A.
    // Each callback releases its session's permit while it waits, so both requests finish.
    // For lifecycle actions, each request waits for the other session's preparation,
    // so the request that closes the cycle throws instead, and the other one finishes.
    @Test(.timeLimit(.minutes(1)), arguments: [false, true], [false, true])
    func independentCallbacksCanRequestEachOther(lifecycle: Bool, streaming: Bool) async throws {
        let bothStarted = Barrier(count: 2)
        let sessions = Locked<[String: LanguageModelSession]>([:])
        let calls = Locked<Set<String>>([])
        func makeSession(_ name: String, other: String) -> LanguageModelSession {
            @Sendable func requestOther() async throws {
                guard calls.withLock({ $0.insert(name).inserted }) else { return }
                await bothStarted.arrive()
                guard let session = sessions.withLock({ $0[other] }) else { return }
                try await send("Nested from \(name)", to: session, streaming: streaming)
            }
            return LanguageModelSession(
                profile: LanguageModelSession.Profile { Instructions(name) }
                    .model(ReentrancyModel())
                    .onPrompt { _ in
                        if lifecycle { try await requestOther() }
                    }
                    .onResponse { _ in
                        if !lifecycle { try await requestOther() }
                    }
            )
        }
        let a = makeSession("A", other: "B")
        let b = makeSession("B", other: "A")
        sessions.withLock { $0 = ["A": a, "B": b] }
        defer { sessions.withLock { $0 = [:] } }

        let outcome = await outcome(within: 30) {
            let first = Task { try await send("Initial", to: a, streaming: streaming) }
            let second = Task { try await send("Initial", to: b, streaming: streaming) }
            return [await first.result, await second.result]
        }

        let results = try #require(try outcome?.get(), "The requests didn't finish")
        let errors = results.compactMap { result -> (any Error)? in
            if case .failure(let error) = result { return error }
            return nil
        }
        if lifecycle {
            #expect(errors.count == 1)
            #expect(
                errors.first as? LanguageModelSession.DynamicProfileError
                    == .requestDuringLifecycleAction
            )
        } else {
            #expect(errors.isEmpty)
            #expect(a.properties.history.count == 4)
            #expect(b.properties.history.count == 4)
        }
    }

    // A task that `onActivate` starts inherits the preparation of the action.
    // After the action ends, that task's request on the same session doesn't wait for it,
    // so the request must run instead of failing.
    @Test(.timeLimit(.minutes(1)))
    func inheritedPreparationExpiresAfterLifecycleAction() async throws {
        let (release, releaseContinuation) = AsyncStream<Void>.makeStream()
        let session = Locked<LanguageModelSession?>(nil)
        let followUp = Locked<Task<Void, any Error>?>(nil)
        let outer = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Deferred request") }
                .model(ReentrancyModel())
                .onActivate {
                    let task = Task {
                        for await _ in release { break }
                        guard let session = session.withLock({ $0 }) else { return }
                        _ = try await session.respond(to: "After activation")
                    }
                    followUp.withLock { $0 = task }
                }
        )
        session.withLock { $0 = outer }
        defer { session.withLock { $0 = nil } }

        _ = try await outer.respond(to: "Initial")
        releaseContinuation.yield()
        releaseContinuation.finish()
        let task = try #require(followUp.withLock { $0 })
        try await task.value
        #expect(outer.properties.history.count == 4)
    }

    @Test func requestDuringLifecycleActionHasAMessage() {
        let error = LanguageModelSession.DynamicProfileError.requestDuringLifecycleAction
        #expect(error.errorDescription?.contains("lifecycle action") == true)
    }
}

private func text(of prompt: Transcript.Prompt) -> String {
    prompt.segments.compactMap { segment -> String? in
        if case .text(let text) = segment { return text.content }
        return nil
    }.joined()
}

private func send(_ prompt: String, to session: LanguageModelSession, streaming: Bool) async throws {
    if streaming {
        _ = try await session.streamResponse(to: prompt).collect()
    } else {
        _ = try await session.respond(to: prompt)
    }
}

/// Suspends each task that arrives until `count` tasks arrive.
private actor Barrier {
    private let count: Int
    private var arrived = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(count: Int) { self.count = count }

    func arrive() async {
        arrived += 1
        if arrived >= count {
            for waiter in waiters { waiter.resume() }
            waiters = []
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }
}

/// Answers each prompt with "response".
private struct ReentrancyModel: LanguageModel {
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
