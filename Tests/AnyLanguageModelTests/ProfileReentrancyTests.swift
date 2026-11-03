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

/// Returns the result of `operation`,
/// or `nil` if it doesn't finish within `seconds`, as when it deadlocks.
private func outcome(
    within seconds: UInt64,
    _ operation: @escaping @Sendable () async throws -> Void
) async -> Result<Void, any Error>? {
    let resumed = Locked(false)
    let tasks = Locked<[Task<Void, Never>]>([])
    let outcome = await withCheckedContinuation {
        (continuation: CheckedContinuation<Result<Void, any Error>?, Never>) in
        @Sendable func finish(_ outcome: Result<Void, any Error>?) {
            let isFirst = resumed.withLock { resumed in
                defer { resumed = true }
                return !resumed
            }
            if isFirst { continuation.resume(returning: outcome) }
        }
        let work = Task {
            do {
                try await operation()
                finish(.success(()))
            } catch {
                finish(.failure(error))
            }
        }
        let timeout = Task {
            try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            finish(nil)
        }
        tasks.withLock { $0 = [work, timeout] }
    }
    // A deadlocked operation can't finish, so leave it; stop the timer.
    tasks.withLock { $0 }.last?.cancel()
    return outcome
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
