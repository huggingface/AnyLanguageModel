import Testing

@testable import AnyLanguageModel

@Suite("Dynamic Profile Composition")
struct DynamicProfileBuilderTests {
    @Test func resolvesInstructionsToolsAndModifiers() {
        let profile = LanguageModelSession.AnyDynamicProfile(
            LanguageModelSession.Profile {
                Instructions("Use this profile")
                ProfileBuilderTool()
            }
            .model(MockLanguageModel())
            .temperature(0.2)
            .samplingMode(.greedy)
            .maximumResponseTokens(32)
            .reasoningLevel(.light)
            .toolCallingMode(.allowed)
            .transcriptErrorHandlingPolicy(nil)
        ).resolveForRequest()
        #expect(profile.model is MockLanguageModel)
        #expect(profile.instructions?.description == "Use this profile")
        #expect(profile.tools.map(\.name) == ["builder"])
        #expect(profile.temperature == 0.2)
        #expect(profile.hasTemperature)
        #expect(profile.samplingMode == .greedy)
        #expect(profile.maximumResponseTokens == 32)
        #expect(profile.reasoningLevel == .light)
        #expect(profile.toolCallingMode == .allowed)
        #expect(profile.hasTranscriptErrorHandlingPolicy)
    }

    @Test func conditionalAndErasedProfilesRetainBranchIdentity() {
        let first = LanguageModelSession.AnyDynamicProfile(ConditionalBuilderProfile(first: true))
            .resolveForRequest()
        let second = LanguageModelSession.AnyDynamicProfile(
            LanguageModelSession.AnyDynamicProfile(ConditionalBuilderProfile(first: false))
        ).resolveForRequest()
        #expect(first.instructions?.description == "First")
        #expect(second.instructions?.description == "Second")
        #expect(first.identity != second.identity)
    }

    @Test func customModifierComposesWithBuiltinModifiers() {
        let profile = LanguageModelSession.AnyDynamicProfile(
            LanguageModelSession.Profile { Instructions("Custom") }
                .modifier(BuilderModifier())
                .maximumResponseTokens(64)
        ).resolveForRequest()
        #expect(profile.temperature == 0.4)
        #expect(profile.maximumResponseTokens == 64)
    }

    @Test func historyTransformsAndCallbacksKeepCompositionOrder() async throws {
        let events = Locked<[String]>([])
        let profile = LanguageModelSession.AnyDynamicProfile(
            LanguageModelSession.Profile { Instructions("Callbacks") }
                .historyTransform { entries in
                    events.withLock { $0.append("inner transform") }
                    return entries
                }
                .onPrompt { _ in events.withLock { $0.append("inner prompt") } }
                .onActivate { events.withLock { $0.append("inner activate") } }
                .onDeactivate { events.withLock { $0.append("inner deactivate") } }
                .historyTransform { entries in
                    events.withLock { $0.append("outer transform") }
                    return entries
                }
                .onPrompt { _ in events.withLock { $0.append("outer prompt") } }
                .onActivate { events.withLock { $0.append("outer activate") } }
                .onDeactivate { events.withLock { $0.append("outer deactivate") } }
        ).resolveForRequest()
        var history: [Transcript.Entry] = []
        for transform in profile.historyTransforms { history = transform(history) }
        for action in profile.onActivate { await action() }
        let prompt = Transcript.Prompt(segments: [.text(.init(content: "Hello"))])
        for action in profile.onPrompt { try await action(prompt) }
        for action in profile.onDeactivate { await action() }
        #expect(
            events.withLock { $0 } == [
                "outer transform", "inner transform",
                "outer activate", "inner activate",
                "outer prompt", "inner prompt",
                "inner deactivate", "outer deactivate",
            ]
        )
    }
}

private struct ConditionalBuilderProfile: LanguageModelSession.DynamicProfile {
    let first: Bool

    var body: some LanguageModelSession.DynamicProfile {
        if first {
            Profile { Instructions("First") }
        } else {
            Profile { Instructions("Second") }
        }
    }
}

private struct BuilderModifier: LanguageModelSession.DynamicProfileModifier {
    func body(content: Content) -> some DynamicProfile {
        content.temperature(0.4)
    }
}

private struct ProfileBuilderTool: Tool {
    let name = "builder"
    let description = "Builder fixture"
    @Generable struct Arguments {}

    func call(arguments: Arguments) async throws -> String { "done" }
}
