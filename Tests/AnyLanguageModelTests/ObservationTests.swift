import Observation
import Testing

@testable import AnyLanguageModel

@Suite("Observation")
struct ObservationTests {
    @Test("Tracking transcript fires onChange when respond mutates it")
    func transcriptObservationTriggeredByRespond() async throws {
        let session = LanguageModelSession(model: MockLanguageModel.fixed("Hello"))
        let changed = Locked(false)

        withObservationTracking {
            _ = session.transcript
        } onChange: {
            changed.withLock { $0 = true }
        }

        try await session.respond(to: "Hi")
        #expect(changed.withLock { $0 } == true)
    }

    @Test("Tracking transcript fires onChange when streamResponse mutates it")
    func transcriptObservationTriggeredByStreamResponse() async throws {
        let session = LanguageModelSession(model: MockLanguageModel.streamingMock())
        let changed = Locked(false)

        withObservationTracking {
            _ = session.transcript
        } onChange: {
            changed.withLock { $0 = true }
        }

        let stream = session.streamResponse(to: "Hi")
        for try await _ in stream {}
        try await Task.sleep(for: .milliseconds(10))

        #expect(changed.withLock { $0 } == true)
    }

    @Test("Tracking isResponding fires onChange when respond mutates it")
    func isRespondingObservationTriggeredByRespond() async throws {
        let session = LanguageModelSession(model: MockLanguageModel.fixed("Hello"))
        let changed = Locked(false)

        withObservationTracking {
            _ = session.isResponding
        } onChange: {
            changed.withLock { $0 = true }
        }

        try await session.respond(to: "Hi")
        #expect(changed.withLock { $0 } == true)
    }

    @Test("No onChange fires when no properties are tracked")
    func noObservationWithoutTrackedRead() async throws {
        let session = LanguageModelSession(model: MockLanguageModel.fixed("Hello"))
        let changed = Locked(false)

        withObservationTracking {
            // Intentionally do not read any session properties
        } onChange: {
            changed.withLock { $0 = true }
        }

        try await session.respond(to: "Hi")
        #expect(changed.withLock { $0 } == false)
    }

    @Test("Re-registering observation tracks subsequent changes")
    func observationReregistersAfterChange() async throws {
        let session = LanguageModelSession(model: MockLanguageModel.fixed("Hello"))
        let changeCount = Locked(0)

        withObservationTracking {
            _ = session.transcript
        } onChange: {
            changeCount.withLock { $0 += 1 }
        }

        try await session.respond(to: "First")
        #expect(changeCount.withLock { $0 } == 1)

        // Re-register after the first change fires
        withObservationTracking {
            _ = session.transcript
        } onChange: {
            changeCount.withLock { $0 += 1 }
        }

        try await session.respond(to: "Second")
        #expect(changeCount.withLock { $0 } == 2)
    }
}
