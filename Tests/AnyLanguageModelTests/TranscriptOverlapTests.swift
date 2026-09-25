import Foundation
import Observation
import Testing

@testable import AnyLanguageModel

@Suite("Overlapping transcript requests")
struct TranscriptOverlapTests {
    @Test(arguments: [
        (false, false, false), (false, false, true), (false, true, false), (false, true, true),
        (true, false, false), (true, false, true), (true, true, false), (true, true, true),
    ])
    func rollbackKeepsOtherRequest(scenario: (Bool, Bool, Bool)) async throws {
        let (streaming, failingStartsFirst, failingFinishesFirst) = scenario
        let failing = ControlledRequest(fails: true)
        let successful = ControlledRequest(fails: false)
        let session = LanguageModelSession(model: OverlapModel(failing: failing, successful: successful))
        session.transcriptErrorHandlingPolicy = .revertTranscript
        func start(_ prompt: String) -> Task<Void, any Error> {
            Task {
                if streaming {
                    _ = try await session.streamResponse(to: prompt).collect()
                } else {
                    _ = try await session.respond(to: prompt)
                }
            }
        }
        let first = start(failingStartsFirst ? "Fail" : "Keep")
        await (failingStartsFirst ? failing : successful).entered.wait()
        let second = start(failingStartsFirst ? "Keep" : "Fail")
        await (failingStartsFirst ? successful : failing).entered.wait()
        let failedTask = failingStartsFirst ? first : second
        let keptTask = failingStartsFirst ? second : first
        if failingFinishesFirst {
            await failing.release.open()
            _ = await failedTask.result
            await successful.release.open()
            try await keptTask.value
        } else {
            await successful.release.open()
            try await keptTask.value
            await failing.release.open()
            _ = await failedTask.result
        }
        await session.waitForResponseCompletion()
        #expect(!session.isResponding)
        #expect(session.transcript.count == 4)
        #expect(
            session.transcript.contains {
                if case .prompt(let p) = $0 { p.segments.first?.description == "Keep" } else { false }
            }
        )
        #expect(
            session.transcript.contains {
                if case .response(let r) = $0 { r.segments.first?.description == "Kept answer" } else { false }
            }
        )
        #expect(session.transcript.contains { $0.id == "kept-calls" })
        #expect(session.transcript.contains { $0.id == "kept-output" })
    }

    @Test func multimodalRollbackKeepsOverlappingResponse() async throws {
        let failing = ControlledRequest(fails: true)
        let successful = ControlledRequest(fails: false)
        let session = LanguageModelSession(model: OverlapModel(failing: failing, successful: successful))
        session.transcriptErrorHandlingPolicy = .revertTranscript
        let failure = Task { try await session.respond(to: "Fail", images: [], generating: String.self) }
        await failing.entered.wait()
        let kept = Task { try await session.respond(to: "Keep") }
        await successful.entered.wait()
        await successful.release.open()
        _ = try await kept.value
        await failing.release.open()
        _ = await failure.result
        #expect(session.transcript.count == 4)
        #expect(session.transcript.contains { $0.id == "kept-output" })
    }

    @Test func cancelledOlderRelayKeepsCompletedNewerRequest() async throws {
        let older = ControlledRequest(fails: true)
        let newer = ControlledRequest(fails: false)
        let session = LanguageModelSession(model: OverlapModel(failing: older, successful: newer))
        session.transcriptErrorHandlingPolicy = .revertTranscript
        let first = Task { try await session.streamResponse(to: "Fail").collect() }
        await older.entered.wait()
        let second = Task { try await session.streamResponse(to: "Keep").collect() }
        await newer.entered.wait()
        await newer.release.open()
        _ = try await second.value
        first.cancel()
        _ = await first.result
        await session.waitForResponseCompletion()
        #expect(!session.isResponding)
        #expect(session.transcript.count == 4)
        #expect(session.transcript.contains { $0.id == "kept-output" })
        await older.release.open()
    }

    @Test func policyChangesNotifyObservation() {
        let session = LanguageModelSession(
            model: OverlapModel(failing: .init(fails: true), successful: .init(fails: false))
        )
        let changes = Locked(0)
        withObservationTracking {
            #expect(session.transcriptErrorHandlingPolicy == nil)
        } onChange: {
            changes.withLock { $0 += 1 }
        }
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        #expect(changes.withLock { $0 } == 1)
        withObservationTracking {
            #expect(session.transcriptErrorHandlingPolicy == .preserveTranscript)
        } onChange: {
            changes.withLock { $0 += 1 }
        }
        session.transcriptErrorHandlingPolicy = .revertTranscript
        #expect(changes.withLock { $0 } == 2)
    }

    @Test func completionWaitIncludesOlderStillRunningRelay() async throws {
        let older = ControlledRequest(fails: true)
        let newer = ControlledRequest(fails: false)
        let session = LanguageModelSession(model: OverlapModel(failing: older, successful: newer))
        session.transcriptErrorHandlingPolicy = .revertTranscript
        let first = Task { try await session.streamResponse(to: "Fail").collect() }
        await older.entered.wait()
        let second = Task { try await session.streamResponse(to: "Keep").collect() }
        await newer.entered.wait()
        await newer.release.open()
        _ = try await second.value
        // The newer relay is already complete, but the older one still owns an active prompt.
        #expect(session.isResponding)
        let waiter = Task {
            await session.waitForResponseCompletion()
            #expect(!session.isResponding)
            #expect(session.transcript.count == 4)
        }
        await older.release.open()
        _ = await first.result
        await waiter.value
    }
}

private actor RequestGate {
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

private struct ControlledRequest: Sendable {
    let fails: Bool
    let entered = RequestGate()
    let release = RequestGate()
}

private enum OverlapError: Error { case expected }

private struct OverlapModel: LanguageModel {
    typealias UnavailableReason = Never
    let failing: ControlledRequest
    let successful: ControlledRequest

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        let request = prompt.description == "Fail" ? failing : successful
        await request.entered.open()
        await request.release.wait()
        if request.fails { throw OverlapError.expected }
        let raw = GeneratedContent("Kept answer")
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: Self.entries)
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let request = prompt.description == "Fail" ? failing : successful
        return .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    await request.entered.open()
                    await request.release.wait()
                    do {
                        if request.fails { throw OverlapError.expected }
                        let raw = GeneratedContent("Kept answer")
                        continuation.yield(
                            .init(
                                content: try Content(raw).asPartiallyGenerated(),
                                rawContent: raw,
                                transcriptEntries: Self.entries
                            )
                        )
                        continuation.finish()
                    } catch { continuation.finish(throwing: error) }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        )
    }

    private static var entries: ArraySlice<Transcript.Entry> {
        [
            .toolCalls(
                .init(
                    id: "kept-calls",
                    [.init(id: "kept-output", toolName: "fixture", arguments: GeneratedContent("{}"))]
                )
            ),
            .toolOutput(.init(id: "kept-output", toolName: "fixture", segments: [.text(.init(content: "Done"))])),
        ]
    }
}
