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
                    let text = prompt.segments.compactMap { segment -> String? in
                        if case .text(let text) = segment { return text.content }
                        return nil
                    }
                    guard text.joined() == "Outer" else { return }
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

    @Test func requestDuringLifecycleActionHasAMessage() {
        let error = LanguageModelSession.DynamicProfileError.requestDuringLifecycleAction
        #expect(error.errorDescription?.contains("lifecycle action") == true)
    }
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
