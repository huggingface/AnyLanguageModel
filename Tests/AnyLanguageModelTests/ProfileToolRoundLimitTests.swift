import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

@Test func ordinarySessionToolRoundLimitUsesRequestOptions() throws {
    let session = LanguageModelSession(model: MockLanguageModel())
    var options = GenerationOptions()
    options.maximumToolRounds = 2
    let budget = session.toolRoundLimit(for: "fixture", options: options)
    try budget.record(completed: [])
    try budget.record(completed: [])
    #expect(throws: LanguageModelSession.ToolRoundLimitExceeded.self) {
        try budget.record(completed: [])
    }
}

@Test func profileToolRoundLimitSharesOriginalCallSiteBudget() async throws {
    let session = LanguageModelSession(
        profile: LanguageModelSession.Profile { Instructions("Budget") }
            .model(ToolBudgetProbeModel())
    )
    var options = GenerationOptions()
    options.maximumToolRounds = 2
    do {
        _ = try await session.respond(to: "Use the configured budget", options: options)
        Issue.record("Expected the third round to exceed the shared budget")
    } catch let error as LanguageModelSession.ToolRoundLimitExceeded {
        #expect(error.limit == 2)
        #expect(error.provider == "first")
        #expect(error.transcriptEntries.count == 1)
    }
}

private struct ToolBudgetProbeModel: LanguageModel {
    typealias UnavailableReason = Never
    var availability: Availability<Never> { .available }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let first = session.toolRoundLimit(for: "first", options: .init())
        let second = session.toolRoundLimit(for: "second", options: .init())
        #expect(first === second)
        try first.record(completed: [])
        try second.record(completed: [])
        let completed: [Transcript.Entry] = [.response(.init(assetIDs: [], segments: []))]
        try second.record(completed: completed)
        throw CancellationError()
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        fatalError("ToolBudgetProbeModel only supports nonstreaming tests")
    }
}
