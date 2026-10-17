import Foundation
import Observation
import Testing

#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS) && !os(watchOS)
    import class FoundationModels.SystemLanguageModel
#endif

@_spi(Compatibility) @testable import AnyLanguageModel

extension SessionPropertyValues {
    @SessionPropertyEntry var profileRevision = 0
}

@Suite("Dynamic profiles")
struct DynamicProfileTests {
    @Test func erasedConcreteProfilesKeepDistinctLifecycleIdentities() async throws {
        let state = LifecycleFixtureState()
        let session = LanguageModelSession(profile: ErasedRoutingProfile(state: state))
        _ = try await session.respond(to: "First")
        _ = try await session.respond(to: "Same profile")
        state.useSecond.withLock { $0 = true }
        _ = try await session.respond(to: "Switch")

        #expect(state.events.withLock { $0 } == ["activate:A", "deactivate:A", "activate:B"])
    }

    @Test func lifecycleHistoryEditsAccumulateAcrossPreparationStages() async throws {
        let state = LifecycleFixtureState()
        let first = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "First old entry"))]))
        let second = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Second old entry"))]))
        let seen = Locked<[[String]]>([])
        let session = LanguageModelSession(
            profile: HistoryEditingLifecycleProfile(state: state, first: first.id, second: second.id, seen: seen),
            history: [first, second]
        )
        _ = try await session.respond(to: "Activate A")
        state.useSecond.withLock { $0 = true }
        _ = try await session.respond(to: "Activate B")

        let stages = seen.withLock { $0 }
        #expect(stages.count == 2)
        #expect(stages.allSatisfy { !$0.contains(first.id) })
        #expect(stages.last?.contains(second.id) == false)
        #expect(!session.properties.history.contains { $0.id == first.id || $0.id == second.id })
    }

    @Test(arguments: [false, true])
    func completionEntryCallbacksShareHistoryEdits(streaming: Bool) async throws {
        let earlier = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Remove me"))]))
        let marker = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Marker"))]))
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Evolving callback history") }
                .model(LifecycleFixtureModel())
                .onReasoning { _ in
                    let properties = try #require(SessionPropertyBinding.values)
                    properties.history.removeAll { $0.id == earlier.id }
                }
                .onResponse { _ in
                    let properties = try #require(SessionPropertyBinding.values)
                    #expect(!properties.history.contains { $0.id == earlier.id })
                    properties.history.append(marker)
                },
            history: [earlier]
        )
        if streaming {
            _ = try await session.streamResponse(to: "Complete").collect()
        } else {
            _ = try await session.respond(to: "Complete")
        }

        #expect(!session.properties.history.contains { $0.id == earlier.id })
        #expect(session.properties.history.last == marker)
    }

    @Test(arguments: [false, true])
    func prewarmUsesActiveModelAndFeedbackUsesProducingModel(streaming: Bool) async throws {
        let state = ProfileFixtureState()
        let session = LanguageModelSession(
            profile: FixtureProfile(
                state: state,
                first: ProfileFixtureModel(id: "a", state: state),
                second: ProfileFixtureModel(id: "b", state: state)
            )
        )
        session.prewarm(promptPrefix: Prompt("Prefix A"))
        if streaming {
            _ = try await session.streamResponse(to: "First").collect()
        } else {
            _ = try await session.respond(to: "First")
        }
        state.selectSecond()
        session.prewarm(promptPrefix: Prompt("Prefix B"))
        #expect(session.logFeedbackAttachment(sentiment: .positive) == Data("a".utf8))
        if streaming {
            _ = try await session.streamResponse(to: "Second").collect()
        } else {
            _ = try await session.respond(to: "Second")
        }
        #expect(session.logFeedbackAttachment(sentiment: .positive) == Data("b".utf8))
        #expect(state.prewarms.withLock { $0 } == ["a:Prefix A", "b:Prefix B"])
        #expect(state.feedback.withLock { $0 } == ["a", "b"])
    }

    @Test func sameSessionReevaluatesModelInstructionsOptionsAndProperties() async throws {
        let state = ProfileFixtureState()
        let first = ProfileFixtureModel(id: "model-a", state: state)
        let second = ProfileFixtureModel(id: "model-b", state: state)
        let session = LanguageModelSession(
            profile: FixtureProfile(state: state, first: first, second: second)
        )

        _ = try await session.respond(to: "First")
        state.selectSecond()
        session.properties.profileRevision = 7
        _ = try await session.respond(to: "Second")

        let requests = state.requests.withLock { $0 }
        #expect(requests.map(\.modelID) == ["model-a", "model-b"])
        #expect(requests.map(\.instructions) == ["Profile A revision 0", "Profile B revision 7"])
        #expect(requests.map(\.temperature) == [0.2, 0.8])
        #expect(requests.map(\.maximumResponseTokens) == [64, 128])
        #expect(requests.map(\.reasoningLevel) == [.light, .deep])
        #expect(session.transcript.count == 4)
        #expect(!session.transcript.contains { if case .instructions = $0 { true } else { false } })
    }

    @Test func callSiteOptionsOverrideProfileDefaults() async throws {
        let state = ProfileFixtureState()
        let model = ProfileFixtureModel(id: "model", state: state)
        let session = LanguageModelSession(
            profile: FixtureProfile(state: state, first: model, second: model)
        )

        _ = try await session.respond(
            to: "Override",
            options: .init(
                temperature: 0.45,
                maximumResponseTokens: 17,
                toolCallingMode: .required
            )
        )

        let request = try #require(state.requests.withLock { $0.last })
        #expect(request.temperature == 0.45)
        #expect(request.maximumResponseTokens == 17)
        #expect(request.toolCallingMode == .required)
        #expect(request.reasoningLevel == .light)
    }

    @Test func lifecycleCallbacksAccumulateInOrderForResponseAndStream() async throws {
        let state = LifecycleFixtureState()
        let model = LifecycleFixtureModel()
        let session = LanguageModelSession(profile: LifecycleFixtureProfile(state: state, model: model))

        _ = try await session.respond(to: "One")
        _ = try await session.streamResponse(to: "Two").collect()
        state.useSecond.withLock { $0 = true }
        _ = try await session.respond(to: "Three")

        #expect(
            state.events.withLock { $0 } == [
                "outer-activate:A", "inner-activate:A", "outer-prompt:A", "inner-prompt:A",
                "outer-reasoning:A", "inner-reasoning:A", "outer-response:A", "inner-response:A",
                "outer-prompt:A", "inner-prompt:A", "outer-reasoning:A", "inner-reasoning:A",
                "outer-response:A", "inner-response:A", "inner-deactivate:A", "outer-deactivate:A",
                "outer-activate:B", "inner-activate:B", "outer-prompt:B", "inner-prompt:B",
                "outer-reasoning:B", "inner-reasoning:B", "outer-response:B", "inner-response:B",
            ]
        )
    }

    @Test func concurrentSnapshotsDispatchEachCallbackOnce() async throws {
        let probe = ConcurrentCallbackProbe()
        let model = LifecycleFixtureModel()
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Concurrent callback") }
                .model(model)
                .onResponse { _ in await probe.recordInvocation() }
        )
        let context = try await session.resolvedRequestContext(including: [], options: .init())
        let entry = Transcript.Entry.response(
            .init(assetIDs: [], segments: [.text(.init(content: "response"))])
        )

        let first = Task {
            try await session.profileDidProduceEntries(
                [entry],
                requestContext: context,
                currentRoundEntries: [entry]
            )
        }
        await probe.waitUntilFirstInvocation()
        let second = Task {
            try await session.profileDidProduceEntries(
                [entry],
                requestContext: context,
                currentRoundEntries: [entry]
            )
        }

        try await second.value
        let invocationCount = await probe.invocationCount
        await probe.releaseFirstInvocation()
        try await first.value

        #expect(invocationCount == 1)
    }

    @Test(arguments: [false, true])
    func completionCallbacksSeeAndPreserveCommittedHistory(streaming: Bool) async throws {
        let histories = Locked<[[Transcript.Entry]]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Completion history") }
                .model(LifecycleFixtureModel())
                .onResponse { _ in
                    let properties = try #require(SessionPropertyBinding.values)
                    let history = properties.history
                    histories.withLock { $0.append(history) }
                    properties.history = history
                }
        )

        if streaming {
            _ = try await session.streamResponse(to: "Complete").collect()
        } else {
            _ = try await session.respond(to: "Complete")
        }

        let history = try #require(histories.withLock { $0.first })
        #expect(history == Array(session.transcript))
        #expect(history.contains { if case .response = $0 { true } else { false } })
        #expect(history.contains { if case .reasoning = $0 { true } else { false } })
    }

    @Test(arguments: [false, true])
    func historyObserversSeeCompletionAndCallbackRollback(streaming: Bool) async throws {
        let completionObserved = Locked(false)
        let rollbackObserved = Locked(false)
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Observe completion") }
                .model(LifecycleFixtureModel())
                .onPrompt { _ in
                    let properties = try #require(SessionPropertyBinding.values)
                    withObservationTracking {
                        _ = properties.history
                    } onChange: {
                        completionObserved.withLock { $0 = true }
                    }
                }
                .onResponse { _ in
                    let properties = try #require(SessionPropertyBinding.values)
                    withObservationTracking {
                        _ = properties.history
                    } onChange: {
                        rollbackObserved.withLock { $0 = true }
                    }
                    throw LifecycleFixtureError.rejected
                }
        )

        await #expect(throws: LifecycleFixtureError.self) {
            if streaming {
                _ = try await session.streamResponse(to: "Roll back").collect()
            } else {
                _ = try await session.respond(to: "Roll back")
            }
        }

        #expect(completionObserved.withLock { $0 })
        #expect(rollbackObserved.withLock { $0 })
        #expect(session.transcript.count == 1)
    }

    @Test func nestedInstructionsReceiveReadOnlyHistory() {
        let writability = Locked<[Bool]>([])
        let session = LanguageModelSession(profile: HistoryBindingProfile(writability: writability))

        _ = session.resolvedRequestContext()

        #expect(writability.withLock { $0 } == [true, false])
    }

    @Test func toolHistoryExcludesStaticInstructions() async {
        let session = LanguageModelSession(model: LifecycleFixtureModel(), instructions: "Static")
        let context = session.resolvedRequestContext()
        let history = await session.withProfileToolExecution(requestContext: context, currentRoundEntries: []) {
            session.properties.history
        }

        #expect(history.isEmpty)
        #expect(session.transcript.count == 1)
    }

    @Test(arguments: [false, true])
    func promptCallbackUsesTheActualRequestAfterHistoryMutation(streaming: Bool) async throws {
        let prompts = Locked<[String]>([])
        let session = LanguageModelSession(profile: PromptRemovingProfile(prompts: prompts))

        if streaming {
            _ = try await session.streamResponse(to: "Actual request").collect()
        } else {
            _ = try await session.respond(to: "Actual request")
        }

        #expect(prompts.withLock { $0 } == ["Actual request"])
    }

    @Test func concurrentStreamsKeepTheirOwnPromptCallbacks() async throws {
        let prompts = Locked<[String]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("Concurrent prompts") }
                .model(LifecycleFixtureModel())
                .onPrompt { (prompt: Transcript.Prompt) in
                    let text = prompt.segments.compactMap { segment in
                        if case .text(let text) = segment { return text.content }
                        return nil
                    }.joined()
                    prompts.withLock { $0.append(text) }
                }
        )
        let first = session.streamResponse(to: "First")
        let second = session.streamResponse(to: "Second")
        async let firstResponse = first.collect()
        async let secondResponse = second.collect()
        _ = try await (firstResponse, secondResponse)

        #expect(prompts.withLock { $0.sorted() } == ["First", "Second"])
    }

    #if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS) && !os(watchOS)
        @Test func nativeSessionEvaluatesOrdinaryDynamicInstructionsOnce() {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let session = LanguageModelSession(
                model: LifecycleFixtureModel(),
                dynamicInstructions: CountingNativeInstructions()
            )

            _ = makeFoundationModelsSession(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: Prompt("Current")
            )
            #expect(session.properties.profileRevision == 1)
            _ = FoundationModelsDynamicInstructionsAdapter(session: session).body
            #expect(session.properties.profileRevision == 2)
        }

        @Test func nativeSessionUsesProjectedHistory() throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let earlier = Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Earlier"))]))
            let prompt = Prompt("Current")
            let current = Transcript.Entry.prompt(
                .init(id: "current", segments: try prompt.makeTranscriptSegments(occurrenceID: "current"))
            )
            let session = LanguageModelSession(
                profile: LanguageModelSession.Profile { Instructions("Projected") }
                    .model(LifecycleFixtureModel())
                    .historyTransform { $0.filter { $0.id != earlier.id } },
                history: [earlier, current]
            )

            let native = makeFoundationModelsSession(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: prompt
            )

            #expect(!native.transcript.contains { if case .prompt = $0 { true } else { false } })
            #expect(Array(session.transcript) == [earlier, current])
        }
    #endif

    @Test(arguments: [false, true])
    func callbackErrorsPropagateWithoutCommittingAResponse(streaming: Bool) async throws {
        let model = LifecycleFixtureModel()
        let session = LanguageModelSession(profile: ThrowingResponseProfile(model: model))

        if streaming {
            await #expect(throws: LifecycleFixtureError.self) {
                _ = try await session.streamResponse(to: "Fail").collect()
            }
        } else {
            await #expect(throws: LifecycleFixtureError.self) {
                _ = try await session.respond(to: "Fail")
            }
        }

        #expect(session.transcript.count == 1)
        #expect(session.transcript.allSatisfy { if case .prompt = $0 { true } else { false } })
    }

    @Test(arguments: [false, true])
    func promptFailureUsesTheProducingRequestPolicy(streaming: Bool) async throws {
        let state = PromptFailureProfileState()
        let session = LanguageModelSession(
            profile: PromptFailureProfile(state: state, model: LifecycleFixtureModel())
        )

        if streaming {
            await #expect(throws: LifecycleFixtureError.self) {
                _ = try await session.streamResponse(to: "Revert").collect()
            }
        } else {
            await #expect(throws: LifecycleFixtureError.self) {
                _ = try await session.respond(to: "Revert")
            }
        }
        #expect(session.transcript.isEmpty)
        #expect(session.transcriptErrorHandlingPolicy == nil)

        state.useRevert.withLock { $0 = false }
        if streaming {
            await #expect(throws: LifecycleFixtureError.self) {
                _ = try await session.streamResponse(to: "Keep").collect()
            }
        } else {
            await #expect(throws: LifecycleFixtureError.self) {
                _ = try await session.respond(to: "Keep")
            }
        }
        #expect(session.transcript.count == 1)
        guard case .prompt = session.transcript[0] else {
            Issue.record("Expected the second request prompt to be retained")
            return
        }
        #expect(session.transcriptErrorHandlingPolicy == nil)
    }

    @Test func dynamicInstructionsAndToolsShareSessionProperties() async throws {
        let model = ProfileFixtureModel(id: "model", state: ProfileFixtureState())
        let session = LanguageModelSession(
            model: model,
            dynamicInstructions: PropertyInstructions()
        )
        session.properties.profileRevision = 4

        let context = session.resolvedRequestContext()
        #expect(context.instructions?.description == "Revision 4")

        let tool = PropertyFixtureTool()
        _ = try await session.withProfileToolExecution(
            requestContext: context,
            currentRoundEntries: []
        ) {
            try await tool.makeOutputSegments(from: GeneratedContent(properties: [:]))
        }
        #expect(session.properties.profileRevision == 5)
    }

    @Test func rehydrationKeepsCanonicalHistoryAndUsesTheCurrentProfile() async throws {
        let oldState = ProfileFixtureState()
        let model = ProfileFixtureModel(id: "model", state: oldState)
        let oldSession = LanguageModelSession(
            profile: FixtureProfile(state: oldState, first: model, second: model)
        )
        _ = try await oldSession.respond(to: "Old")
        let saved = oldSession.transcript

        let restoredState = ProfileFixtureState()
        restoredState.selectSecond()
        let restoredModel = ProfileFixtureModel(id: "restored", state: restoredState)
        let restored = LanguageModelSession(
            profile: FixtureProfile(
                state: restoredState,
                first: restoredModel,
                second: restoredModel
            ),
            history: saved
        )
        _ = try await restored.respond(to: "New")

        #expect(restoredState.requests.withLock { $0.last?.instructions } == "Profile B revision 0")
        #expect(restored.transcript.count == saved.count + 2)
        #expect(!restored.transcript.contains { if case .instructions = $0 { true } else { false } })
    }

    @Test func transformedCurrentRoundDoesNotReplaceCanonicalTranscript() async throws {
        let state = ProfileFixtureState()
        let model = ProfileFixtureModel(id: "model-a", state: state)
        let session = LanguageModelSession(
            profile: FixtureProfile(state: state, first: model, second: model)
        )
        _ = try await session.respond(to: "First")
        let durableBefore = session.transcript
        let call = Transcript.ToolCall(
            id: "call-1",
            toolName: "fixture",
            arguments: GeneratedContent(properties: [:])
        )
        let output = Transcript.ToolOutput(
            id: call.id,
            toolName: call.toolName,
            segments: [.text(.init(content: "output"))]
        )
        let inFlight: [Transcript.Entry] = [
            .toolCalls(.init([call])),
            .toolOutput(output),
        ]

        let context = try await session.resolvedRequestContext(
            including: inFlight,
            options: .init()
        )

        #expect(context.transcript.contains { $0.id == output.id })
        #expect(state.transformedHistory.withLock { $0.last?.contains(output.id) } == true)
        #expect(session.transcript == durableBefore)
    }

    @Test func toolCallbacksUseProducingSnapshotAndSessionPropertiesStayIsolated() async throws {
        let state = ProfileFixtureState()
        let model = ProfileFixtureModel(id: "model-a", state: state)
        let first = LanguageModelSession(
            profile: FixtureProfile(state: state, first: model, second: model)
        )
        let second = LanguageModelSession(
            profile: FixtureProfile(state: state, first: model, second: model)
        )
        let call = Transcript.ToolCall(
            id: "call",
            toolName: "fixture",
            arguments: GeneratedContent(properties: [:])
        )
        let output = Transcript.ToolOutput(
            id: call.id,
            toolName: call.toolName,
            segments: [.text(.init(content: "done"))]
        )
        let request = try await first.resolvedRequestContext(including: [], options: .init())
        let unrelated = Transcript.Entry.prompt(
            .init(id: "unrelated", segments: [.text(.init(content: "Other request"))])
        )
        first.properties.history = [unrelated]
        let callEntry = Transcript.Entry.toolCalls(.init([call]))
        let outputEntry = Transcript.Entry.toolOutput(output)

        try await first.profileWillExecuteToolCall(
            call,
            requestContext: request,
            currentRoundEntries: [callEntry]
        )
        try await first.profileDidProduceToolOutput(
            for: call,
            output: output,
            requestContext: request,
            currentRoundEntries: [callEntry, outputEntry]
        )
        let toolHistory = await first.withProfileToolExecution(
            requestContext: request,
            currentRoundEntries: [callEntry, outputEntry]
        ) {
            first.properties.history.map(\.id)
        }

        #expect(first.properties.profileRevision == 1)
        #expect(second.properties.profileRevision == 0)
        #expect(state.callbackEvents.withLock { $0 } == ["call:call", "output:call"])
        #expect(
            state.callbackHistories.withLock { $0 }
                == [[callEntry.id], [callEntry.id, outputEntry.id]]
        )
        #expect(toolHistory == [callEntry.id, outputEntry.id])
    }
}

private struct FixtureProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let state: ProfileFixtureState
    let first: ProfileFixtureModel
    let second: ProfileFixtureModel
    @SessionProperty(\.profileRevision) private var revision
    @SessionProperty(\.history) private var history

    var body: some LanguageModelSession.DynamicProfile {
        if state.usesSecond {
            LanguageModelSession.Profile {
                Instructions("Profile B revision \(revision)")
                FixtureProfileTool()
            }
            .model(second)
            .temperature(0.8)
            .maximumResponseTokens(128)
            .reasoningLevel(.deep)
            .historyTransform(recordingTransform)
            .onToolCall(perform: recordCall)
            .onToolOutput(perform: recordOutput)
        } else {
            LanguageModelSession.Profile {
                Instructions("Profile A revision \(revision)")
                FixtureProfileTool()
            }
            .model(first)
            .temperature(0.2)
            .maximumResponseTokens(64)
            .reasoningLevel(.light)
            .historyTransform(recordingTransform)
            .onToolCall(perform: recordCall)
            .onToolOutput(perform: recordOutput)
        }
    }

    private func recordingTransform(_ entries: [Transcript.Entry]) -> [Transcript.Entry] {
        state.transformedHistory.withLock { $0.append(entries.map(\.id)) }
        return entries
    }

    private func recordCall(_ call: Transcript.ToolCall) async throws {
        state.callbackEvents.withLock { $0.append("call:\(call.id)") }
        state.callbackHistories.withLock { $0.append(history.map(\.id)) }
    }

    private func recordOutput(
        _ call: Transcript.ToolCall,
        _ output: Transcript.ToolOutput
    ) async throws {
        revision += 1
        state.callbackEvents.withLock { $0.append("output:\(output.id)") }
        state.callbackHistories.withLock { $0.append(history.map(\.id)) }
    }
}

private struct FixtureProfileTool: Tool {
    let name = "fixture"
    let description = "Fixture"

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String { "done" }
}

private struct ProfileFixtureModel: LanguageModel {
    typealias UnavailableReason = Never
    let id: String
    let state: ProfileFixtureState

    func prewarm(for session: LanguageModelSession, promptPrefix: Prompt?) {
        state.prewarms.withLock { $0.append("\(id):\(promptPrefix?.description ?? "")") }
    }

    func logFeedbackAttachment(
        within session: LanguageModelSession,
        sentiment: LanguageModelFeedback.Sentiment?,
        issues: [LanguageModelFeedback.Issue],
        desiredOutput: Transcript.Entry?
    ) -> Data {
        state.feedback.withLock { $0.append(id) }
        return Data(id.utf8)
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let context = try await session.resolvedRequestContext(including: [], options: options)
        state.requests.withLock {
            $0.append(
                .init(
                    modelID: id,
                    instructions: context.instructions?.description,
                    temperature: context.options.temperature,
                    maximumResponseTokens: context.options.maximumResponseTokens,
                    toolCallingMode: context.options.toolCallingMode,
                    reasoningLevel: context.contextOptions.reasoningLevel
                )
            )
        }
        let raw = GeneratedContent("response")
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [])
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            let task = Task {
                do {
                    let response: LanguageModelSession.Response<Content> = try await respond(
                        within: session,
                        to: prompt,
                        generating: type,
                        includeSchemaInPrompt: includeSchemaInPrompt,
                        options: options
                    )
                    continuation.yield(
                        .init(
                            content: response.content.asPartiallyGenerated(),
                            rawContent: response.rawContent
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return .init(stream: stream)
    }
}

private final class ProfileFixtureState: @unchecked Sendable {
    struct Request: Equatable {
        let modelID: String
        let instructions: String?
        let temperature: Double?
        let maximumResponseTokens: Int?
        let toolCallingMode: GenerationOptions.ToolCallingMode?
        let reasoningLevel: ContextOptions.ReasoningLevel?
    }

    private let selection = Locked(false)
    let requests = Locked<[Request]>([])
    let transformedHistory = Locked<[[String]]>([])
    let callbackEvents = Locked<[String]>([])
    let callbackHistories = Locked<[[String]]>([])
    let prewarms = Locked<[String]>([])
    let feedback = Locked<[String]>([])

    var usesSecond: Bool { selection.withLock { $0 } }
    func selectSecond() { selection.withLock { $0 = true } }
}

private struct PropertyInstructions: DynamicInstructions, @unchecked Sendable {
    @SessionProperty(\.profileRevision) private var revision

    var body: some DynamicInstructions {
        Instructions("Revision \(revision)")
    }
}

private struct PropertyFixtureTool: Tool, @unchecked Sendable {
    let name = "property"
    let description = "Updates a session property"
    @SessionProperty(\.profileRevision) private var revision

    @Generable struct Arguments {}

    func call(arguments: Arguments) async throws -> String {
        revision += 1
        return "updated"
    }
}

private enum LifecycleFixtureError: Error {
    case rejected
}

private struct HistoryBindingProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let writability: Locked<[Bool]>

    var body: some LanguageModelSession.DynamicProfile {
        writability.withLock { $0.append(SessionPropertyBinding.history?.isWritable ?? false) }
        return LanguageModelSession.Profile { HistoryBindingInstructions(writability: writability) }
            .model(LifecycleFixtureModel())
    }
}

private struct HistoryBindingInstructions: DynamicInstructions, @unchecked Sendable {
    let writability: Locked<[Bool]>

    var body: some DynamicInstructions {
        writability.withLock { $0.append(SessionPropertyBinding.history?.isWritable ?? true) }
        return Instructions("Read-only instructions")
    }
}

private struct PromptRemovingProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let prompts: Locked<[String]>
    @SessionProperty(\.history) private var history

    var body: some LanguageModelSession.DynamicProfile {
        history = []
        return LanguageModelSession.Profile { Instructions("Removed prompt") }
            .model(LifecycleFixtureModel())
            .onPrompt { (prompt: Transcript.Prompt) in
                let text = prompt.segments.compactMap { segment in
                    if case .text(let text) = segment { return text.content }
                    return nil
                }.joined()
                prompts.withLock { $0.append(text) }
            }
    }
}

private actor ConcurrentCallbackProbe {
    private(set) var invocationCount = 0
    private var firstInvocationWaiter: CheckedContinuation<Void, Never>?
    private var firstInvocationRelease: CheckedContinuation<Void, Never>?

    func recordInvocation() async {
        invocationCount += 1
        guard invocationCount == 1 else { return }
        firstInvocationWaiter?.resume()
        firstInvocationWaiter = nil
        await withCheckedContinuation { firstInvocationRelease = $0 }
    }

    func waitUntilFirstInvocation() async {
        guard invocationCount == 0 else { return }
        await withCheckedContinuation { firstInvocationWaiter = $0 }
    }

    func releaseFirstInvocation() {
        firstInvocationRelease?.resume()
        firstInvocationRelease = nil
    }
}

private final class LifecycleFixtureState: @unchecked Sendable {
    let useSecond = Locked(false)
    let events = Locked<[String]>([])

    func record(_ value: String) {
        events.withLock { $0.append(value) }
    }
}

private enum FirstErasedProfile {}
private enum SecondErasedProfile {}

private struct ErasedLeafProfile<Identity>: LanguageModelSession.DynamicProfile {
    var body: some LanguageModelSession.DynamicProfile {
        LanguageModelSession.Profile { Instructions("Same leaf type") }
    }
}

private struct ErasedRoutingProfile: LanguageModelSession.DynamicProfile {
    let state: LifecycleFixtureState

    var body: some LanguageModelSession.DynamicProfile {
        let second = state.useSecond.withLock { $0 }
        let profile =
            second
            ? LanguageModelSession.AnyDynamicProfile(ErasedLeafProfile<SecondErasedProfile>())
            : LanguageModelSession.AnyDynamicProfile(ErasedLeafProfile<FirstErasedProfile>())
        let label = second ? "B" : "A"
        return profile.model(LifecycleFixtureModel())
            .onActivate { state.record("activate:\(label)") }
            .onDeactivate { state.record("deactivate:\(label)") }
    }
}

private struct HistoryEditingLifecycleProfile: LanguageModelSession.DynamicProfile {
    let state: LifecycleFixtureState
    let first: String
    let second: String
    let seen: Locked<[[String]]>

    var body: some LanguageModelSession.DynamicProfile {
        if state.useSecond.withLock({ $0 }) {
            LanguageModelSession.Profile { Instructions("B") }
                .model(LifecycleFixtureModel())
                .onActivate {
                    guard let properties = SessionPropertyBinding.values else { return }
                    seen.withLock { $0.append(properties.history.map(\.id)) }
                    properties.history.removeAll { $0.id == second }
                }
                .onPrompt { _ in
                    let properties = try #require(SessionPropertyBinding.values)
                    seen.withLock { $0.append(properties.history.map(\.id)) }
                }
        } else {
            LanguageModelSession.Profile { Instructions("A") }
                .model(LifecycleFixtureModel())
                .onDeactivate {
                    guard let properties = SessionPropertyBinding.values else { return }
                    properties.history.removeAll { $0.id == first }
                }
        }
    }
}

private struct CountingNativeInstructions: DynamicInstructions {
    @SessionProperty(\.profileRevision) private var revision

    var body: some DynamicInstructions {
        revision += 1
        return Instructions("Revision \(revision)")
    }
}

private struct LifecycleFixtureProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let state: LifecycleFixtureState
    let model: LifecycleFixtureModel

    var body: some LanguageModelSession.DynamicProfile {
        if state.useSecond.withLock({ $0 }) {
            profile(label: "B")
        } else {
            profile(label: "A")
        }
    }

    private func profile(label: String) -> some LanguageModelSession.DynamicProfile {
        LanguageModelSession.Profile { Instructions("Lifecycle \(label)") }
            .model(model)
            .onActivate { state.record("inner-activate:\(label)") }
            .onDeactivate { state.record("inner-deactivate:\(label)") }
            .onPrompt { state.record("inner-prompt:\(label)") }
            .onReasoning { state.record("inner-reasoning:\(label)") }
            .onResponse { state.record("inner-response:\(label)") }
            .onActivate { state.record("outer-activate:\(label)") }
            .onDeactivate { state.record("outer-deactivate:\(label)") }
            .onPrompt { state.record("outer-prompt:\(label)") }
            .onReasoning { state.record("outer-reasoning:\(label)") }
            .onResponse { state.record("outer-response:\(label)") }
    }
}

private struct ThrowingResponseProfile: LanguageModelSession.DynamicProfile {
    let model: LifecycleFixtureModel

    var body: some LanguageModelSession.DynamicProfile {
        LanguageModelSession.Profile { Instructions("Throw after response") }
            .model(model)
            .onResponse { throw LifecycleFixtureError.rejected }
    }
}

private final class PromptFailureProfileState: @unchecked Sendable {
    let useRevert = Locked(true)
}

private struct PromptFailureProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let state: PromptFailureProfileState
    let model: LifecycleFixtureModel

    var body: some LanguageModelSession.DynamicProfile {
        if state.useRevert.withLock({ $0 }) {
            LanguageModelSession.Profile { Instructions("Revert prompt failures") }
                .model(model)
                .transcriptErrorHandlingPolicy(.revertTranscript)
                .onPrompt { throw LifecycleFixtureError.rejected }
        } else {
            LanguageModelSession.Profile { Instructions("Keep prompt failures") }
                .model(model)
                .onPrompt { throw LifecycleFixtureError.rejected }
        }
    }
}

private struct LifecycleFixtureModel: LanguageModel {
    typealias UnavailableReason = Never

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let raw = GeneratedContent("response")
        let reasoning = Transcript.Entry.reasoning(
            .init(segments: [.text(.init(content: "reasoning"))])
        )
        return .init(
            content: try Content(raw),
            rawContent: raw,
            transcriptEntries: [reasoning]
        )
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            let task = Task {
                do {
                    let response: LanguageModelSession.Response<Content> = try await respond(
                        within: session,
                        to: prompt,
                        generating: type,
                        includeSchemaInPrompt: includeSchemaInPrompt,
                        options: options
                    )
                    continuation.yield(
                        .init(
                            content: response.content.asPartiallyGenerated(),
                            rawContent: response.rawContent,
                            transcriptEntries: response.transcriptEntries
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return .init(stream: stream)
    }
}
