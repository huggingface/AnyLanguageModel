import Foundation
import Observation
import Testing

@testable import AnyLanguageModel

extension SessionPropertyValues {
    @SessionPropertyEntry var revision = 0
    @SessionPropertyEntry var optionalSelection: String? = nil
    @SessionPropertyEntry var items: [Int] = []
    @SessionPropertyEntry var mode: SessionPropertyMode = .first
}

enum SessionPropertyMode {
    case first
    case second
}

@Suite("Session properties")
struct SessionPropertyTests {
    @Test func dynamicInstructionsReadAndUpdateSessionProperties() {
        let session = LanguageModelSession(
            model: SessionPropertyFixtureModel(),
            dynamicInstructions: SessionPropertyFixtureInstructions()
        )

        #expect(session.properties.revision == 0)
        let first = session.resolvedRequestContext()
        let second = session.resolvedRequestContext()

        #expect(first.instructions?.description == "Revision 0")
        #expect(second.instructions?.description == "Revision 1")
        #expect(session.properties.revision == 2)
    }

    @Test func historyIsInstructionFreeAndUsesTheCurrentRequestSnapshot() {
        let prompt = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(.init(content: "Earlier"))])
        )
        let session = LanguageModelSession(
            model: SessionPropertyFixtureModel(),
            dynamicInstructions: SessionPropertyHistoryInstructions(),
            history: [prompt]
        )

        let context = session.resolvedRequestContext()

        #expect(context.instructions?.description == "History count 1")
        #expect(context.transcript.count == 2)
        #expect(session.properties.history == [prompt])
    }

    @Test func requestUsesOneHistorySnapshotDuringDynamicResolution() async {
        let initial = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(.init(content: "Initial"))])
        )
        let replacement = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(.init(content: "Replacement"))])
        )
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        let resume = DispatchSemaphore(value: 0)
        let session = LanguageModelSession(
            model: SessionPropertyFixtureModel(),
            dynamicInstructions: BlockingHistoryInstructions(
                started: startedContinuation,
                resume: resume
            ),
            history: [initial]
        )

        let request = Task.detached { session.resolvedRequestContext() }
        for await _ in started.prefix(1) {}
        session.properties.history = [replacement]
        resume.signal()
        let context = await request.value

        #expect(context.instructions?.description == "History count 1")
        #expect(
            context.transcript.filter { entry in
                if case .instructions = entry { return false }
                return true
            } == [initial]
        )
        #expect(session.properties.history == [replacement])
    }

    @Test func replacingHistoryNotifiesTranscriptObservers() {
        let session = LanguageModelSession(model: SessionPropertyFixtureModel())
        let changed = Locked(false)
        let prompt = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(.init(content: "Replacement"))])
        )

        withObservationTracking {
            _ = session.transcript
        } onChange: {
            changed.withLock { $0 = true }
        }
        session.properties.history = [prompt]

        #expect(changed.withLock { $0 })
    }

    @Test func generatedPropertiesParticipateInObservation() {
        let session = LanguageModelSession(model: SessionPropertyFixtureModel())
        let changed = Locked(false)

        withObservationTracking {
            _ = session.properties.revision
        } onChange: {
            changed.withLock { $0 = true }
        }
        session.properties.revision = 1

        #expect(changed.withLock { $0 })
    }

    @Test func generatedPropertiesPreserveExplicitTypeContext() {
        let session = LanguageModelSession(model: SessionPropertyFixtureModel())

        #expect(session.properties.optionalSelection == nil)
        #expect(session.properties.items.isEmpty)
        #expect(session.properties.mode == .first)
        session.properties.optionalSelection = "selected"
        session.properties.items = [1, 2]
        session.properties.mode = .second
        #expect(session.properties.optionalSelection == "selected")
        #expect(session.properties.items == [1, 2])
        #expect(session.properties.mode == .second)
    }

    @Test func historyObservationTracksSessionTranscriptChanges() async throws {
        let session = LanguageModelSession(model: CompletingSessionPropertyModel())
        let changed = Locked(false)

        withObservationTracking {
            _ = session.properties.history
        } onChange: {
            changed.withLock { $0 = true }
        }
        _ = try await session.respond(to: "Observe history")

        #expect(changed.withLock { $0 })
        #expect(session.properties.history.count == 2)
    }

    @Test func propertiesAndHistoryAreIsolatedBetweenSessions() {
        let first = LanguageModelSession(model: SessionPropertyFixtureModel())
        let second = LanguageModelSession(model: SessionPropertyFixtureModel())
        let prompt = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(.init(content: "Replacement"))])
        )

        first.properties.revision = 7
        first.properties.history = [prompt]

        #expect(first.properties.revision == 7)
        #expect(first.transcript == Transcript(entries: [prompt]))
        #expect(second.properties.revision == 0)
        #expect(second.transcript.isEmpty)
        let _: any Observable = first.properties
    }

    @Test func replacingHistoryPreservesStaticInstructions() {
        let session = LanguageModelSession(
            model: SessionPropertyFixtureModel(),
            instructions: "Keep these instructions"
        )
        let prompt = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(.init(content: "Replacement"))])
        )

        session.properties.history = [prompt]

        #expect(session.properties.history == [prompt])
        #expect(session.transcript.count == 2)
        guard case .instructions = session.transcript[0] else {
            Issue.record("Expected the static instructions entry to remain first")
            return
        }
        #expect(session.transcript[1] == prompt)
    }
}

private struct SessionPropertyFixtureInstructions: DynamicInstructions {
    @SessionProperty(\.revision) private var revision

    var body: some DynamicInstructions {
        let currentRevision = revision
        revision += 1
        return Instructions("Revision \(currentRevision)")
    }
}

private struct SessionPropertyHistoryInstructions: DynamicInstructions {
    @SessionProperty(\.history) private var history

    var body: some DynamicInstructions {
        Instructions("History count \(history.count)")
    }
}

private struct BlockingHistoryInstructions: DynamicInstructions {
    @SessionProperty(\.history) private var history
    let started: AsyncStream<Void>.Continuation
    let resume: DispatchSemaphore

    var body: some DynamicInstructions {
        let count = history.count
        started.yield()
        resume.wait()
        return Instructions("History count \(count)")
    }
}

private struct SessionPropertyFixtureModel: LanguageModel {
    typealias UnavailableReason = Never

    var availability: Availability<Never> { .available }

    func supports(locale: Locale) -> Bool { true }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        fatalError("SessionPropertyFixtureModel does not generate responses")
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        fatalError("SessionPropertyFixtureModel does not generate responses")
    }
}

private struct CompletingSessionPropertyModel: LanguageModel {
    typealias UnavailableReason = Never

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let raw = GeneratedContent("Done")
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [])
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        fatalError("This fixture only supports non-streaming responses")
    }
}
