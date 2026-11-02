import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

extension SessionPropertyValues {
    @SessionPropertyEntry var ownershipResolutionCount = 0
}

@Suite("Profile request ownership")
struct ProfileRequestOwnershipTests {
    @Test(arguments: [false, true])
    func initialRequestResolvesTheProfileOnce(streaming: Bool) async throws {
        let session = LanguageModelSession(profile: CountingProfile(model: OwnershipModel()))
        // Creating the session evaluates the body once to record its instructions.
        #expect(session.properties.ownershipResolutionCount == 1)

        if streaming {
            _ = try await session.streamResponse(to: "Count").collect()
        } else {
            _ = try await session.respond(to: "Count")
        }

        #expect(session.properties.ownershipResolutionCount == 2)
        #expect(session.transcript.count == 3)
        #expect(session.properties.history.count == 2)
    }

    @Test(arguments: [false, true])
    func rollbackKeepsAnEarlierOutputWithTheSameCallID(streaming: Bool) async throws {
        let rejects = Locked(false)
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Reused call IDs") }
                .model(OwnershipModel(toolRound: true))
                .onResponse { _ in
                    if rejects.withLock({ $0 }) { throw OwnershipError.rejected }
                }
        )

        _ = try await session.respond(to: "First")
        let first = Array(session.transcript)
        rejects.withLock { $0 = true }
        await #expect(throws: OwnershipError.self) {
            if streaming {
                _ = try await session.streamResponse(to: "Second").collect()
            } else {
                _ = try await session.respond(to: "Second")
            }
        }

        #expect(Array(session.transcript.prefix(first.count)) == first)
        #expect(session.transcript.count == first.count + 1)
    }

    @Test(arguments: [false, true])
    func rollbackRemovesAnOutputWhoseToolCallsTheCallbackRemoved(streaming: Bool) async throws {
        let rejects = Locked(false)
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Reused call IDs") }
                .model(OwnershipModel(toolRound: true))
                .onResponse { _ in
                    guard rejects.withLock({ $0 }) else { return }
                    let properties = try #require(SessionPropertyBinding.values)
                    let toolCalls = properties.history.last { entry in
                        if case .toolCalls = entry { return true }
                        return false
                    }
                    properties.history.removeAll { $0.id == toolCalls?.id }
                    throw OwnershipError.rejected
                }
        )

        _ = try await session.respond(to: "First")
        let first = Array(session.transcript)
        rejects.withLock { $0 = true }
        await #expect(throws: OwnershipError.self) {
            if streaming {
                _ = try await session.streamResponse(to: "Second").collect()
            } else {
                _ = try await session.respond(to: "Second")
            }
        }

        #expect(Array(session.transcript.prefix(first.count)) == first)
        #expect(session.transcript.count == first.count + 1)
    }

    @Test func historyWritesKeepEarlierOutputsThatMatchAPendingOutput() throws {
        let session = LanguageModelSession(model: OwnershipModel())
        let call = Transcript.ToolCall(id: "call_0", toolName: "fixture", arguments: GeneratedContent(properties: [:]))
        let output = Transcript.Entry.toolOutput(
            .init(id: call.id, toolName: call.toolName, segments: [.text(.init(id: "segment", content: "done"))])
        )
        let durable: [Transcript.Entry] = [
            .prompt(.init(segments: [.text(.init(content: "Prompt"))])),
            .toolCalls(.init(id: "round-1", [call])),
            output,
        ]
        // The pending round reuses the call ID and produces an identical output.
        let entries = durable + [.toolCalls(.init(id: "round-2", [call])), output]
        let pendingKeys = Set(keyedTranscriptEntries(entries).suffix(2).map(\.key))
        let binding = session.properties.historyBinding(
            entries,
            isWritable: true,
            protecting: pendingKeys,
            persistingAt: Locked(session.historyVersion)
        )

        binding.entries = durable
        binding.persist(binding.entries)

        #expect(Array(session.transcript) == durable)
        #expect(binding.entries == entries)
    }

    @Test func editedPendingEntriesDoNotBecomeDurable() throws {
        let session = LanguageModelSession(model: OwnershipModel())
        let call = Transcript.ToolCall(id: "call_0", toolName: "fixture", arguments: GeneratedContent(properties: [:]))
        let durable: [Transcript.Entry] = [.prompt(.init(segments: [.text(.init(content: "Prompt"))]))]
        let pending: [Transcript.Entry] = [
            .toolCalls(.init(id: "round-1", [call])),
            .toolOutput(.init(id: call.id, toolName: call.toolName, segments: [.text(.init(content: "done"))])),
        ]
        let entries = durable + pending
        let pendingKeys = Set(keyedTranscriptEntries(entries).suffix(2).map(\.key))
        let binding = session.properties.historyBinding(
            entries,
            isWritable: true,
            protecting: pendingKeys,
            persistingAt: Locked(session.historyVersion)
        )
        let edited = Transcript.Entry.toolOutput(
            .init(id: call.id, toolName: call.toolName, segments: [.text(.init(content: "edited"))])
        )

        binding.entries = durable + [pending[0], edited]
        binding.persist(binding.entries)

        #expect(Array(session.transcript) == durable)
        #expect(binding.entries == entries)
    }

    @Test func cancellingAResponseCancelsItsPromptHook() async throws {
        let started = AsyncStream<Void>.makeStream()
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Slow prompt hook") }
                .model(OwnershipModel())
                .onPrompt { _ in
                    started.continuation.yield()
                    try await Task.sleep(for: .seconds(10))
                }
        )
        let clock = ContinuousClock()
        let start = clock.now

        let response = Task { try await session.respond(to: "Cancel me") }
        var iterator = started.stream.makeAsyncIterator()
        _ = await iterator.next()
        response.cancel()

        await #expect(throws: CancellationError.self) { _ = try await response.value }
        #expect(clock.now - start < .seconds(5))
    }

    @Test func cancelledQueuedPreparationRunsNoPromptHook() async throws {
        let gate = OwnershipGate()
        let prompts = Locked<[String]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Queued prompt hooks") }
                .model(OwnershipModel())
                .onPrompt { (prompt: Transcript.Prompt) in
                    let text = prompt.segments.compactMap { segment in
                        if case .text(let text) = segment { return text.content }
                        return nil
                    }.joined()
                    prompts.withLock { $0.append(text) }
                    if text == "First" { await gate.wait() }
                }
        )

        let first = Task { try await session.respond(to: "First") }
        await gate.waitUntilWaiting()
        let second = Task { try await session.respond(to: "Second") }
        // Let the second request queue behind the first preparation.
        try await Task.sleep(for: .milliseconds(100))
        second.cancel()
        // The queued preparation still waits for its predecessor, which isn't cancelled.
        await gate.open()

        _ = try await first.value
        await #expect(throws: CancellationError.self) { _ = try await second.value }
        #expect(prompts.withLock { $0 } == ["First"])
    }
}

private enum OwnershipError: Error { case rejected }

private struct CountingProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let model: OwnershipModel
    @SessionProperty(\.ownershipResolutionCount) private var count

    var body: some LanguageModelSession.DynamicProfile {
        count += 1
        return LanguageModelSession.Profile { Instructions("Revision \(count)") }
            .model(model)
    }
}

private actor OwnershipGate {
    private var isOpen = false
    private var waiter: CheckedContinuation<Void, Never>?
    private var watchers: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            waiter = continuation
            for watcher in watchers { watcher.resume() }
            watchers = []
        }
    }

    func waitUntilWaiting() async {
        guard waiter == nil else { return }
        await withCheckedContinuation { watchers.append($0) }
    }

    func open() {
        isOpen = true
        waiter?.resume()
        waiter = nil
    }
}

/// Answers each request with a fixed response,
/// optionally after one Tool round that reuses the call ID `call_0`.
private struct OwnershipModel: LanguageModel {
    typealias UnavailableReason = Never
    var toolRound = false

    private func entries() -> [Transcript.Entry] {
        guard toolRound else { return [] }
        let call = Transcript.ToolCall(id: "call_0", toolName: "fixture", arguments: GeneratedContent(properties: [:]))
        return [
            .toolCalls(.init([call])),
            .toolOutput(.init(id: call.id, toolName: call.toolName, segments: [.text(.init(content: "done"))])),
        ]
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let raw = GeneratedContent("response")
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: ArraySlice(entries()))
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let entries = entries()
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            do {
                let raw = GeneratedContent("response")
                continuation.yield(
                    .init(
                        content: try Content(raw).asPartiallyGenerated(),
                        rawContent: raw,
                        transcriptEntries: ArraySlice(entries)
                    )
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return .init(stream: stream)
    }
}
