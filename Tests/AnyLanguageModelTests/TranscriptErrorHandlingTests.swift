import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Transcript error handling")
struct TranscriptErrorHandlingTests {
    @Test(arguments: [false, true])
    func cancelledPartialAnswerIsNotCommitted(completedTools: Bool) async throws {
        let session = LanguageModelSession(model: CheckpointModel(completedTools: completedTools))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        let consumer = Task {
            for try await _ in session.streamResponse(to: "Question") {
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        _ = await consumer.result
        await session.waitForResponseCompletion()
        #expect(!session.isResponding)
        #expect(session.transcript.count == (completedTools ? 3 : 1))
        #expect(!session.transcript.contains { if case .response = $0 { true } else { false } })
        if completedTools {
            #expect(session.transcript[1].id == "calls")
            #expect(session.transcript[2].id == "call")
        }
    }

    @Test func nilPolicyRetainsPromptWithoutCommittingPartialAnswerOrTools() async throws {
        let session = LanguageModelSession(model: CheckpointModel(completedTools: true))
        #expect(session.transcriptErrorHandlingPolicy == nil)
        let consumer = Task {
            for try await _ in session.streamResponse(to: "Question") {
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        _ = await consumer.result
        await session.waitForResponseCompletion()
        #expect(session.transcript.count == 1)
        guard case .prompt = session.transcript[0] else { Issue.record("Expected prompt only"); return }
    }

    @Test func revertCancellationRestoresPreviousTranscript() async throws {
        let old = Transcript.Entry.prompt(.init(id: "previous", segments: [.text(.init(content: "Previous"))]))
        let session = LanguageModelSession(
            model: CheckpointModel(completedTools: true),
            transcript: Transcript(entries: [old])
        )
        session.transcriptErrorHandlingPolicy = .revertTranscript
        let consumer = Task {
            for try await _ in session.streamResponse(to: "Question") {
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        _ = await consumer.result
        await session.waitForResponseCompletion()
        #expect(Array(session.transcript) == [old])
    }

    @Test(arguments: [TranscriptErrorHandlingPolicy.preserveTranscript, .revertTranscript])
    func thrownFailureUsesLatestCumulativeCheckpoint(policy: TranscriptErrorHandlingPolicy) async throws {
        let session = LanguageModelSession(model: CheckpointModel(completedTools: true, finish: .failure))
        session.transcriptErrorHandlingPolicy = policy
        await #expect(throws: FixtureError.self) {
            for try await _ in session.streamResponse(to: "Question") {}
        }
        await session.waitForResponseCompletion()
        if policy == .preserveTranscript {
            #expect(session.transcript.count == 3)
            #expect(session.transcript.filter { if case .toolOutput = $0 { true } else { false } }.count == 1)
        } else {
            #expect(session.transcript.isEmpty)
        }
        #expect(!session.isResponding)
        #expect(!session.transcript.contains { if case .response = $0 { true } else { false } })
    }

    @Test func nilPolicyThrownFailureRetainsPrompt() async throws {
        let session = LanguageModelSession(model: CheckpointModel(completedTools: true, finish: .failure))
        await #expect(throws: FixtureError.self) {
            for try await _ in session.streamResponse(to: "Question") {}
        }
        #expect(session.transcript.count == 1)
    }

    @Test func nonstreamFailureRevertsPrompt() async throws {
        let previous = Transcript.Entry.prompt(.init(id: "previous", segments: [.text(.init(content: "Previous"))]))
        let session = LanguageModelSession(
            model: CheckpointModel(completedTools: true, finish: .failure),
            transcript: Transcript(entries: [previous])
        )
        session.transcriptErrorHandlingPolicy = .revertTranscript
        await #expect(throws: FixtureError.self) { _ = try await session.respond(to: "Question") }
        #expect(Array(session.transcript) == [previous])
        #expect(!session.isResponding)
    }

    @Test func nonstreamPreserveFailureHasNoCheckpointChannel() async throws {
        let session = LanguageModelSession(model: CheckpointModel(completedTools: true, finish: .failure))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        await #expect(throws: FixtureError.self) { _ = try await session.respond(to: "Question") }
        #expect(session.transcript.count == 1)
    }

    @Test(arguments: [TranscriptErrorHandlingPolicy.preserveTranscript, .revertTranscript])
    func successStillCommitsFullResponseOnce(policy: TranscriptErrorHandlingPolicy) async throws {
        let session = LanguageModelSession(model: CheckpointModel(completedTools: true, finish: .success))
        session.transcriptErrorHandlingPolicy = policy
        let result = try await session.streamResponse(to: "Question").collect()
        await session.waitForResponseCompletion()
        #expect(result.content == "Answer")
        #expect(session.transcript.count == 4)
        #expect(session.transcript.filter { if case .response = $0 { true } else { false } }.count == 1)
        #expect(session.transcript.filter { if case .toolOutput = $0 { true } else { false } }.count == 1)
    }
}

private enum FixtureError: Error { case failed }

private struct CheckpointModel: LanguageModel {
    typealias UnavailableReason = Never
    enum Finish: Sendable { case suspended, failure, success }
    var completedTools: Bool
    var finish: Finish = .suspended

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        try await streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        ).collect()
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        .init(
            stream: AsyncThrowingStream { continuation in
                do {
                    var entries: [Transcript.Entry] = []
                    if completedTools {
                        entries.append(
                            .toolCalls(
                                .init(
                                    id: "calls",
                                    [
                                        .init(id: "call", toolName: "fixture", arguments: GeneratedContent("{}"))
                                    ]
                                )
                            )
                        )
                        entries.append(
                            .toolOutput(
                                .init(id: "call", toolName: "fixture", segments: [.text(.init(content: "Done"))])
                            )
                        )
                    }
                    // Repeated cumulative entries exercise replacement/checkpoint behavior.
                    for text in ["Partial answer", "Partial answer"] {
                        let raw = GeneratedContent(text)
                        continuation.yield(
                            .init(
                                content: try Content(raw).asPartiallyGenerated(),
                                rawContent: raw,
                                transcriptEntries: ArraySlice(entries)
                            )
                        )
                    }
                    switch finish {
                    case .suspended: return
                    case .failure: throw FixtureError.failed
                    case .success:
                        let raw = GeneratedContent("Answer")
                        continuation.yield(
                            .init(
                                content: try Content(raw).asPartiallyGenerated(),
                                rawContent: raw,
                                transcriptEntries: ArraySlice(entries)
                            )
                        )
                        continuation.finish()
                    }
                } catch { continuation.finish(throwing: error) }
            }
        )
    }
}
