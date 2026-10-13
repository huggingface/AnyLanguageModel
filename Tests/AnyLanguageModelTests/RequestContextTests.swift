import Testing

@testable import AnyLanguageModel

@Suite("Request context")
struct RequestContextTests {
    @Test func staticSessionContextMatchesTheSession() {
        let session = LanguageModelSession(
            model: MockLanguageModel(),
            tools: [WeatherTool()],
            instructions: "Static"
        )

        let context = session.resolvedRequestContext()

        #expect(context.instructions?.description == "Static")
        #expect(context.tools.map(\.name) == session.tools.map(\.name))
        #expect(context.transcript == session.transcript)
    }

    @Test func contextReflectsTheCurrentTranscript() async throws {
        let session = LanguageModelSession(model: MockLanguageModel.fixed("Hi"), instructions: "Static")

        _ = try await session.respond(to: "Hello")

        #expect(session.resolvedRequestContext().transcript == session.transcript)
        #expect(session.resolvedRequestContext().transcript.count == 3)
    }
}
