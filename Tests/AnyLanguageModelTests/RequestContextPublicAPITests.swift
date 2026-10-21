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
    #expect(context.tools.first is PublicSessionPropertyTool)
    let _: PublicSessionPropertyTool.SessionProperty<Transcript.HistoryView> = .init(\.history)
}

@Test func requestContextRetainsConcreteToolIdentity() {
    let original = PublicReferenceTool()
    let session = LanguageModelSession(
        model: OpenAILanguageModel(apiKey: "unused", model: "test"),
        tools: [original]
    )
    let context = session.resolvedRequestContext()
    #expect(context.tools.first as? PublicReferenceTool === original)
}

private final class PublicReferenceTool: Tool {
    let name = "reference"
    let description = "Preserve the supplied Tool instance"

    func call(arguments: String) async throws -> String { arguments }
}

private struct PublicSessionPropertyTool: Tool {
    @SessionProperty(\.history) private var history

    let name = "publicSessionProperty"
    let description = "Read session history"

    func call(arguments: String) async throws -> String {
        String(history.count)
    }
}
