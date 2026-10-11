import Foundation
import Observation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

extension SessionPropertyValues {
    @SessionPropertyEntry var profileRevision = 0
}

@Suite("Dynamic profiles")
struct DynamicProfileTests {
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

    @MainActor
    @Test func appleConcurrencyAndObservationSurfaceCompiles() {
        var callbackCount = 0
        _ = LanguageModelSession.Profile { Instructions("Actor callbacks") }
            .model(LifecycleFixtureModel())
            .onActivate { callbackCount += 1 }
            .onDeactivate { callbackCount += 1 }
            .onPrompt { (_: Transcript.Prompt) in callbackCount += 1 }
            .onResponse { (_: Transcript.Response) in callbackCount += 1 }
            .onReasoning { (_: Transcript.Reasoning) in callbackCount += 1 }
            .onToolCall { (_: Transcript.ToolCall) in callbackCount += 1 }
            .onToolOutput { (_: Transcript.ToolCall, _: Transcript.ToolOutput) in
                callbackCount += 1
            }

        let session = LanguageModelSession(model: LifecycleFixtureModel())
        let _: any Observable = session.properties
    }

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

        try await first.profileWillExecuteToolCall(
            call,
            requestContext: request,
            currentRoundEntries: [.toolCalls(.init([call]))]
        )
        try await first.profileDidProduceToolOutput(
            for: call,
            output: output,
            requestContext: request,
            currentRoundEntries: [.toolCalls(.init([call])), .toolOutput(output)]
        )

        #expect(first.properties.profileRevision == 1)
        #expect(second.properties.profileRevision == 0)
        #expect(state.callbackEvents.withLock { $0 } == ["call:call", "output:call"])
    }

    @Test func continuationHistoryMutationDoesNotCommitPendingEntries() async throws {
        let state = ProfileFixtureState()
        let model = ProfileFixtureModel(id: "model", state: state)
        let session = LanguageModelSession(profile: HistoryWritingProfile(model: model))
        _ = try await session.respond(to: "Durable")
        let durableBefore = session.transcript
        let call = Transcript.ToolCall(
            id: "pending-call",
            toolName: "fixture",
            arguments: GeneratedContent(properties: [:])
        )
        let output = Transcript.ToolOutput(
            id: call.id,
            toolName: call.toolName,
            segments: [.text(.init(content: "pending"))]
        )
        let pending: [Transcript.Entry] = [
            .toolCalls(.init([call])),
            .toolOutput(output),
        ]

        let context = try await session.resolvedRequestContext(
            including: pending,
            options: .init()
        )

        #expect(context.transcript.contains { $0.id == output.id })
        #expect(session.transcript == durableBefore)
    }

    @Test func profileErrorPolicyDoesNotLeakIntoTheNextProfile() async throws {
        let state = LifecycleFixtureState()
        let session = LanguageModelSession(
            profile: ErrorPolicyProfile(state: state, model: FailingProfileModel())
        )
        session.transcriptErrorHandlingPolicy = .preserveTranscript

        await #expect(throws: LifecycleFixtureError.self) {
            _ = try await session.respond(to: "Revert")
        }
        #expect(session.transcript.isEmpty)

        state.useSecond.withLock { $0 = true }
        await #expect(throws: LifecycleFixtureError.self) {
            _ = try await session.streamResponse(to: "Preserve").collect()
        }
        #expect(session.transcript.count == 1)
        #expect(session.transcript.allSatisfy { if case .prompt = $0 { true } else { false } })
        #expect(session.transcriptErrorHandlingPolicy == .preserveTranscript)
    }
}

private struct FixtureProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let state: ProfileFixtureState
    let first: ProfileFixtureModel
    let second: ProfileFixtureModel
    @SessionProperty(\.profileRevision) private var revision

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
    }

    private func recordOutput(
        _ call: Transcript.ToolCall,
        _ output: Transcript.ToolOutput
    ) async throws {
        revision += 1
        state.callbackEvents.withLock { $0.append("output:\(output.id)") }
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

private final class LifecycleFixtureState: @unchecked Sendable {
    let useSecond = Locked(false)
    let events = Locked<[String]>([])

    func record(_ value: String) {
        events.withLock { $0.append(value) }
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

private struct HistoryWritingProfile: LanguageModelSession.DynamicProfile {
    let model: ProfileFixtureModel
    @SessionProperty(\.history) private var history

    var body: some LanguageModelSession.DynamicProfile {
        history = history
        return LanguageModelSession.Profile { Instructions("History") }
            .model(model)
    }
}

private struct ErrorPolicyProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
    let state: LifecycleFixtureState
    let model: FailingProfileModel

    var body: some LanguageModelSession.DynamicProfile {
        if state.useSecond.withLock({ $0 }) {
            LanguageModelSession.Profile { Instructions("Caller policy") }
                .model(model)
        } else {
            LanguageModelSession.Profile { Instructions("Profile policy") }
                .model(model)
                .transcriptErrorHandlingPolicy(.revertTranscript)
        }
    }
}

private struct FailingProfileModel: LanguageModel {
    typealias UnavailableReason = Never

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        throw LifecycleFixtureError.rejected
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        .init(stream: AsyncThrowingStream { $0.finish(throwing: LifecycleFixtureError.rejected) })
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
