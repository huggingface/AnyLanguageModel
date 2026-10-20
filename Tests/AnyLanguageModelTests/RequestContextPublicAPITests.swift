import AnyLanguageModel
import Testing

@Test func requestContextRemainsPublicWithoutSPI() {
    let session = LanguageModelSession(
        model: OpenAILanguageModel(apiKey: "unused", model: "test"),
        tools: [PublicSessionPropertyTool()],
        instructions: "Public request context"
    )
    let context: LanguageModelSession.RequestContext = session.resolvedRequestContext()
    #expect(context.transcript.count == session.transcript.count)
    #expect(context.instructions?.description == "Public request context")
    #expect(context.tools.map(\.name) == ["publicSessionProperty"])
    let _: PublicSessionPropertyTool.SessionProperty<Transcript.HistoryView> = .init(\.history)
}

private struct PublicSessionPropertyTool: Tool {
    @SessionProperty(\.history) private var history

    let name = "publicSessionProperty"
    let description = "Read session history"

    func call(arguments: String) async throws -> String {
        String(history.count)
    }
}
