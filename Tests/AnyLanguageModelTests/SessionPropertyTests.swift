import Foundation
import Observation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

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
    @Test(arguments: [false, true])
    func executionToolsCarryPropertiesIntoDetachedExecution(dynamic: Bool) async throws {
        let session =
            dynamic
            ? LanguageModelSession(
                model: SessionPropertyFixtureModel(),
                dynamicInstructions: PropertyToolInstructions()
            )
            : LanguageModelSession(
                model: SessionPropertyFixtureModel(),
                tools: [PropertyBoundTool()],
                instructions: "Static"
            )
        let earlier = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Earlier"))]))
        session.properties.history = [earlier]
        session.properties.revision = 7
        let context = session.resolvedRequestContext()
        #expect(context.tools.first is PropertyBoundTool)
        let tool = try #require(context.toolsForExecution(in: session).first)
        let output = try await Task.detached {
            #expect(SessionPropertyBinding.values == nil)
            return try await tool.makeOutputSegments(from: GeneratedContent("read"))
        }.value

        switch try #require(output.first) {
        case .text(let text):
            #expect(text.content == "revision:7 history:1")
        case .structure(let structured):
            #expect(try String(structured.content) == "revision:7 history:1")
        case .image:
            Issue.record("Unexpected image output")
        }
        #expect(session.properties.revision == 8)
        #expect(Array(session.properties.history) == [earlier])
        #expect(SessionPropertyBinding.values == nil)
    }

    @Test func propertyDefaultsCanReadAnotherProperty() {
        let session = LanguageModelSession(model: SessionPropertyFixtureModel())
        SessionPropertyBinding.$values.withValue(session.properties) {
            #expect(session.properties[DerivedPropertyKey.self] == 42)
            session.properties[BasePropertyKey.self] = 7
            #expect(session.properties[DerivedPropertyKey.self] == 42)
        }
    }

    @Test(arguments: [false, true], [false, true])
    func replacingHistoryDiscardsStaleGenerationCommits(streaming: Bool, images: Bool) async throws {
        let (started, start) = AsyncStream<Void>.makeStream()
        let (resume, resumeContinuation) = AsyncStream<Void>.makeStream()
        let session = LanguageModelSession(
            model: PausedHistoryModel(started: start, resume: resume),
            instructions: "Keep instructions"
        )
        let request = Task {
            if streaming {
                let stream =
                    images
                    ? session.streamResponse(to: "Old request", images: [])
                    : session.streamResponse(to: "Old request")
                return try await stream.collect().content
            }
            let response =
                try await images
                ? session.respond(to: "Old request", images: [])
                : session.respond(to: "Old request")
            return response.content
        }
        for await _ in started.prefix(1) {}
        let replacement = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "New conversation"))]))
        session.properties.history = [replacement]
        resumeContinuation.yield()

        #expect(try await request.value == "Done")
        #expect(Array(session.properties.history) == [replacement])
        #expect(session.transcript.count == 2)
        #expect(!session.isResponding)
    }

    @Test func replacingHistoryDiscardsStaleStreamingFailureCheckpoint() async {
        let (started, start) = AsyncStream<Void>.makeStream()
        let (resume, resumeContinuation) = AsyncStream<Void>.makeStream()
        let session = LanguageModelSession(model: PausedHistoryModel(started: start, resume: resume, fail: true))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        let request = Task { try await session.streamResponse(to: "Old request").collect() }
        for await _ in started.prefix(1) {}
        let replacement = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "New conversation"))]))
        session.properties.history = [replacement]
        resumeContinuation.yield()

        await #expect(throws: PausedHistoryError.self) { try await request.value }
        #expect(Array(session.properties.history) == [replacement])
        #expect(!session.isResponding)
    }

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
        #expect(Array(session.properties.history) == [prompt])
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
        #expect(Array(session.properties.history) == [replacement])
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

        #expect(Array(session.properties.history) == [prompt])
        #expect(session.transcript.count == 2)
        guard case .instructions = session.transcript[0] else {
            Issue.record("Expected the static instructions entry to remain first")
            return
        }
        #expect(session.transcript[1] == prompt)
    }

    @Test func boundHistoryOnlyAppliesToItsOwningSession() {
        let earlier = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Earlier"))]))
        let replacement = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Replacement"))]))
        let other = LanguageModelSession(model: SessionPropertyFixtureModel())
        other.properties.history = [earlier]
        let session = LanguageModelSession(
            model: SessionPropertyFixtureModel(),
            dynamicInstructions: ForeignSessionHistoryInstructions(other: other, replacement: replacement)
        )

        let context = session.resolvedRequestContext()

        #expect(context.instructions?.description == "Other history count 1")
        #expect(session.properties.history.isEmpty)
        #expect(Array(other.properties.history) == [replacement])
    }

    @Test func replacingHistoryRejectsForeignInstructions() {
        let other = LanguageModelSession(model: SessionPropertyFixtureModel(), instructions: "Foreign")
        let prompt = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Earlier"))]))
        other.properties.history = [prompt]
        let session = LanguageModelSession(
            model: SessionPropertyFixtureModel(),
            dynamicInstructions: SessionPropertyHistoryInstructions()
        )

        session.properties.history = .init(other.transcript)
        let context = session.resolvedRequestContext()

        #expect(session.transcript == Transcript(entries: [prompt]))
        #expect(context.instructions?.description == "History count 1")
        #expect(context.transcript.count == 2)
    }
}

private struct PropertyToolInstructions: DynamicInstructions {
    var body: some DynamicInstructions {
        Instructions("Property Tool")
        PropertyBoundTool()
    }
}

private struct PropertyBoundTool: Tool {
    @LanguageModelSession.SessionProperty(\.revision) private var revision
    @LanguageModelSession.SessionProperty(\.history) private var history

    let name = "property"
    let description = "Read and update session properties"
    var parameters: GenerationSchema { String.generationSchema }

    func call(arguments: String) async throws -> String {
        let result = "revision:\(revision) history:\(history.count)"
        revision += 1
        return result
    }
}

private enum BasePropertyKey: SessionPropertyKey {
    static let defaultValue = 42
}

private enum DerivedPropertyKey: SessionPropertyKey {
    static var defaultValue: Int {
        SessionPropertyBinding.values![BasePropertyKey.self]
    }
}

private enum PausedHistoryError: Error { case failed }

private struct PausedHistoryModel: LanguageModel {
    typealias UnavailableReason = Never
    let started: AsyncStream<Void>.Continuation
    let resume: AsyncStream<Void>
    var fail = false

    private var entries: [Transcript.Entry] {
        [.reasoning(.init(segments: [.text(.init(content: "Old reasoning"))]))]
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        started.yield()
        for await _ in resume.prefix(1) {}
        let raw = GeneratedContent("Done")
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: entries[...])
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        .init(
            stream: AsyncThrowingStream { continuation in
                Task {
                    do {
                        let raw = GeneratedContent("Done")
                        continuation.yield(
                            .init(
                                content: try Content.PartiallyGenerated(raw),
                                rawContent: raw,
                                transcriptEntries: entries[...]
                            )
                        )
                        started.yield()
                        for await _ in resume.prefix(1) {}
                        if fail { throw PausedHistoryError.failed }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        )
    }
}

private struct ForeignSessionHistoryInstructions: DynamicInstructions {
    let other: LanguageModelSession
    let replacement: Transcript.Entry

    var body: some DynamicInstructions {
        let count = other.properties.history.count
        other.properties.history = [replacement]
        return Instructions("Other history count \(count)")
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
