import Foundation
import Testing

import AnyLanguageModel

@Suite("Dynamic profile public API")
struct DynamicProfilePublicAPITests {
    @Test func profileInitializerIsPublicWithoutSPI() async throws {
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Public profile") }
                .model(MockLanguageModel { _, _ in "Hello" })
        )

        let response = try await session.respond(to: "Hi")

        #expect(response.content == "Hello")
        #expect(session.transcript.count == 3)
        guard case .instructions(let instructions)? = session.transcript.first else {
            Issue.record("Expected the transcript to start with the profile's instructions")
            return
        }
        #expect(instructions.segments.map(\.description) == ["Public profile"])
    }

    @Test(arguments: [false, true])
    func missingModelThrowsBeforeHooksRunAndHistoryChanges(streaming: Bool) async throws {
        let events = PublicProfileEvents()
        let earlier = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Earlier"))]))
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("No model") }
                .onActivate { events.record("activate") }
                .onPrompt { _ in events.record("prompt") },
            history: [earlier]
        )

        let error = await #expect(throws: LanguageModelSession.DynamicProfileError.missingModel) {
            if streaming {
                _ = try await session.streamResponse(to: "Hi").collect()
            } else {
                _ = try await session.respond(to: "Hi")
            }
        }

        #expect(error?.localizedDescription.contains(".model(_:)") == true)
        #expect(events.values.isEmpty)
        #expect(session.properties.history.map(\.id) == [earlier.id])
        #expect(session.transcript.count == 2)
        #expect(!session.isResponding)
    }
}

private final class PublicProfileEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func record(_ event: String) {
        lock.withLock { storage.append(event) }
    }

    var values: [String] {
        lock.withLock { storage }
    }
}
