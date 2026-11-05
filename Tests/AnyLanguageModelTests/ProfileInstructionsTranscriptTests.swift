import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

/// A profile session records the active profile's instructions
/// in an instructions entry at the start of its transcript,
/// as Foundation Models does.
@Suite("Profile instructions in the transcript")
struct ProfileInstructionsTranscriptTests {
    @Test func creatingASessionRecordsTheInstructionsWithoutRunningActions() {
        let events = Locked<[String]>([])
        let evaluations = Locked(0)
        let session = LanguageModelSession(
            profile: EvaluationCountingProfile(evaluations: evaluations, events: events)
        )

        #expect(evaluations.withLock { $0 } == 1)
        #expect(events.withLock { $0 }.isEmpty)
        #expect(session.transcript.count == 1)
        let instructions = leadingInstructions(session.transcript)
        #expect(instructions.map(texts) == "Counted")
        #expect(instructions?.toolDefinitions.map(\.name) == ["instructions_fixture"])
        #expect(session.properties.history.isEmpty)
    }

    @Test func historyInstructionsKeepTheirIDAndLaterOnesAreLeftOut() {
        let base = Transcript.Instructions(segments: [.text(.init(content: "Base"))], toolDefinitions: [])
        let middle = Transcript.Instructions(segments: [.text(.init(content: "Middle"))], toolDefinitions: [])
        let prompt = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Earlier"))]))
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Current") }
                .model(MockLanguageModel()),
            history: [.instructions(base), prompt, .instructions(middle)]
        )

        #expect(session.transcript.map(\.id) == [base.id, prompt.id])
        #expect(leadingInstructions(session.transcript).map(texts) == "Current")
    }

    @Test func aProfileWithoutInstructionsUsesTheHistoryInstructions() {
        let base = Transcript.Instructions(segments: [.text(.init(content: "Base"))], toolDefinitions: [])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { [any Tool]() }
                .model(MockLanguageModel()),
            history: [.instructions(base)]
        )

        #expect(Array(session.transcript) == [.instructions(base)])
        let context = session.resolvedRequestContext()
        #expect(context.instructions?.description == "Base")
        #expect(leadingInstructions(context.transcript).map(texts) == "Base")
    }

    @Test func aProfileWithoutInstructionsOrHistoryRecordsNoEntry() {
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { [any Tool]() }
                .model(MockLanguageModel())
        )

        #expect(session.transcript.isEmpty)
        let context = session.resolvedRequestContext()
        #expect(context.instructions == nil)
        #expect(context.transcript.isEmpty)
    }

    @Test func eachRequestUpdatesTheEntryWithoutChangingTheHistoryVersion() async throws {
        let detail = Locked("first")
        let session = LanguageModelSession(profile: ChangingInstructionsProfile(detail: detail))
        let id = try #require(session.transcript.first?.id)
        _ = try await session.respond(to: "Hi")
        #expect(leadingInstructions(session.transcript).map(texts) == "Detail first")

        detail.withLock { $0 = "second" }
        // The entry changes when a request resolves the profile, not before.
        #expect(leadingInstructions(session.transcript).map(texts) == "Detail first")
        let version = session.historyVersion
        _ = session.resolvedRequestContext()

        #expect(leadingInstructions(session.transcript).map(texts) == "Detail second")
        #expect(session.transcript.first?.id == id)
        #expect(session.historyVersion == version)
        #expect(session.transcript.count == 3)
    }

    @Test func aMissingModelLeavesTheRecordedEntryUnchanged() async throws {
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("No model") }
        )
        let recorded = Array(session.transcript)
        #expect(leadingInstructions(session.transcript).map(texts) == "No model")

        await #expect(throws: LanguageModelSession.DynamicProfileError.missingModel) {
            _ = try await session.respond(to: "Hi")
        }
        #expect(Array(session.transcript) == recorded)
    }
}

private func leadingInstructions(_ transcript: Transcript) -> Transcript.Instructions? {
    guard case .instructions(let instructions)? = transcript.first else { return nil }
    return instructions
}

private func texts(_ instructions: Transcript.Instructions) -> String {
    instructions.segments.compactMap { segment -> String? in
        if case .text(let text) = segment { return text.content }
        return nil
    }.joined()
}

private struct EvaluationCountingProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let evaluations: Locked<Int>
    let events: Locked<[String]>

    var body: some LanguageModelSession.DynamicProfile {
        evaluations.withLock { $0 += 1 }
        return LanguageModelSession.Profile {
            Instructions("Counted")
            InstructionsFixtureTool()
        }
        .model(MockLanguageModel())
        .onActivate { events.withLock { $0.append("activate") } }
        .onPrompt { _ in events.withLock { $0.append("prompt") } }
    }
}

private struct ChangingInstructionsProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let detail: Locked<String>

    var body: some LanguageModelSession.DynamicProfile {
        let detail = detail.withLock { $0 }
        return LanguageModelSession.Profile { Instructions("Detail \(detail)") }
            .model(MockLanguageModel())
    }
}

private struct InstructionsFixtureTool: Tool {
    let name = "instructions_fixture"
    let description = "Fixture"

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String { "done" }
}
