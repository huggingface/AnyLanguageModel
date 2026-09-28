import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Dynamic instructions")
struct DynamicInstructionsTests {
    @Test func bodyReevaluatesForEveryNonstreamingRequest() async throws {
        let state = DynamicFixtureState()
        let model = DynamicContextModel(state: state, continuesAfterTool: false)
        let session = LanguageModelSession(
            model: model,
            dynamicInstructions: FixtureDynamicInstructions(state: state)
        )

        #expect(state.evaluationCount == 0)
        _ = try await session.respond(to: "First")
        state.select(.b)
        _ = try await session.respond(to: "Second")

        #expect(
            model.snapshots.withLock { $0 } == [
                .init(instructions: "Instructions A", tools: ["tool-a"]),
                .init(instructions: "Instructions B", tools: ["tool-b"]),
            ]
        )
        #expect(state.evaluationCount == 2)
    }

    @Test func bodyReevaluatesForEveryStreamingRequest() async throws {
        let state = DynamicFixtureState()
        let model = DynamicContextModel(state: state, continuesAfterTool: false)
        let session = LanguageModelSession(
            model: model,
            dynamicInstructions: FixtureDynamicInstructions(state: state)
        )

        #expect(state.evaluationCount == 0)
        _ = try await session.streamResponse(to: "First").collect()
        state.select(.b)
        _ = try await session.streamResponse(to: "Second").collect()

        #expect(
            model.snapshots.withLock { $0 } == [
                .init(instructions: "Instructions A", tools: ["tool-a"]),
                .init(instructions: "Instructions B", tools: ["tool-b"]),
            ]
        )
        #expect(state.evaluationCount == 2)
    }

    @Test(arguments: [false, true])
    func toolContinuationReevaluatesAndExecutesProducingSnapshot(streaming: Bool) async throws {
        let state = DynamicFixtureState()
        let model = DynamicContextModel(state: state, continuesAfterTool: true)
        let session = LanguageModelSession(
            model: model,
            dynamicInstructions: FixtureDynamicInstructions(state: state)
        )

        if streaming {
            _ = try await session.streamResponse(to: "Use a tool").collect()
        } else {
            _ = try await session.respond(to: "Use a tool")
        }

        #expect(
            model.snapshots.withLock { $0 } == [
                .init(instructions: "Instructions A", tools: ["tool-a"]),
                .init(instructions: "Instructions B", tools: ["tool-b"]),
            ]
        )
        #expect(state.executedTools == ["tool-a"])
        #expect(state.evaluationCount == 2)
        #expect(session.transcript.count == 4)
        guard case .prompt = session.transcript[0],
            case .toolCalls(let calls) = session.transcript[1],
            case .toolOutput(let output) = session.transcript[2],
            case .response = session.transcript[3]
        else {
            Issue.record("Expected prompt, tool call, tool output, and response")
            return
        }
        #expect(calls.first?.id == output.id)
        #expect(output.toolName == "tool-a")
    }

    @Test func historyDoesNotPersistDynamicInstructionsAndRehydratesWithCurrentState() async throws {
        let state = DynamicFixtureState()
        let firstModel = DynamicContextModel(state: state, continuesAfterTool: false)
        let firstSession = LanguageModelSession(
            model: firstModel,
            dynamicInstructions: FixtureDynamicInstructions(state: state)
        )
        _ = try await firstSession.respond(to: "First")

        #expect(
            !firstSession.transcript.contains {
                if case .instructions = $0 { true } else { false }
            }
        )

        state.select(.b)
        let restoredModel = DynamicContextModel(state: state, continuesAfterTool: false)
        let restoredSession = LanguageModelSession(
            model: restoredModel,
            dynamicInstructions: FixtureDynamicInstructions(state: state),
            history: firstSession.transcript
        )
        _ = try await restoredSession.respond(to: "Second")

        #expect(
            restoredModel.snapshots.withLock { $0 } == [
                .init(instructions: "Instructions B", tools: ["tool-b"])
            ]
        )
        #expect(restoredSession.transcript.count == 4)
        #expect(
            !restoredSession.transcript.contains {
                if case .instructions = $0 { true } else { false }
            }
        )
    }

    @Test func builderComposesNestedConditionalEmptyAndToolArrayContent() {
        let state = DynamicFixtureState()
        let enabled = true
        let dynamic = AnyDynamicInstructions(erasing: FixtureComposition(state: state, enabled: enabled))
        let session = LanguageModelSession(
            model: DynamicContextModel(state: state, continuesAfterTool: false),
            dynamicInstructions: dynamic
        )

        let context = session.resolvedRequestContext()

        #expect(context.instructions?.description == "Outer\nNested\nFor each")
        #expect(context.tools.map(\.name) == ["tool-a", "tool-b"])
        #expect(session.transcript.isEmpty)
    }

    @Test func staticSessionRequestContextPreservesExistingBehavior() {
        let state = DynamicFixtureState()
        let tool = FixtureTool(name: "static-tool", state: state)
        let session = LanguageModelSession(
            model: DynamicContextModel(state: state, continuesAfterTool: false),
            tools: [tool],
            instructions: "Static"
        )

        let context = session.resolvedRequestContext()

        #expect(context.instructions?.description == "Static")
        #expect(context.tools.map(\.name) == ["static-tool"])
        #expect(context.transcript == session.transcript)
        #expect(session.instructions?.description == "Static")
        #expect(session.tools.map(\.name) == ["static-tool"])
    }

    @Test func failedResponseDoesNotReplayCompletedDynamicToolSideEffect() async throws {
        let state = DynamicFixtureState()
        let model = FailingAfterToolModel(state: state)
        let session = LanguageModelSession(
            model: model,
            dynamicInstructions: FixtureDynamicInstructions(state: state)
        )

        await #expect(throws: DynamicFixtureError.failed) {
            _ = try await session.respond(to: "Fail after tool")
        }
        _ = try await session.respond(to: "Retry")

        #expect(state.executedTools == ["tool-a"])
        #expect(
            model.snapshots.withLock { $0 } == [
                .init(instructions: "Instructions A", tools: ["tool-a"]),
                .init(instructions: "Instructions B", tools: ["tool-b"]),
            ]
        )
    }

    @Test func cancelledResponseDoesNotReplayCompletedDynamicToolSideEffect() async throws {
        let state = DynamicFixtureState()
        let control = DynamicCancellationControl()
        let model = CancellingAfterToolModel(state: state, control: control)
        let session = LanguageModelSession(
            model: model,
            dynamicInstructions: FixtureDynamicInstructions(state: state)
        )
        var started = control.started.makeAsyncIterator()

        let response = Task {
            try await session.respond(to: "Cancel after tool")
        }
        _ = await started.next()
        response.cancel()

        await #expect(throws: CancellationError.self) {
            _ = try await response.value
        }
        _ = try await session.respond(to: "Retry")

        #expect(state.executedTools == ["tool-a"])
        #expect(
            model.snapshots.withLock { $0 } == [
                .init(instructions: "Instructions A", tools: ["tool-a"]),
                .init(instructions: "Instructions B", tools: ["tool-b"]),
            ]
        )
    }
}

private struct FixtureDynamicInstructions: DynamicInstructions {
    let state: DynamicFixtureState

    var body: some DynamicInstructions {
        let snapshot = state.snapshotForEvaluation()
        Instructions(snapshot.instructions)
        [snapshot.tool]
    }
}

private struct FixtureComposition: DynamicInstructions {
    let state: DynamicFixtureState
    let enabled: Bool

    var body: some DynamicInstructions {
        Instructions("Outer")
        if enabled {
            NestedFixtureInstructions(state: state)
        }
        EmptyDynamicInstructions()
        ForEach([FixtureInstruction(id: 1, text: "For each")]) { item in
            Instructions(item.text)
        }
    }
}

private struct FixtureInstruction: Identifiable {
    let id: Int
    let text: String
}

private struct NestedFixtureInstructions: DynamicInstructions {
    let state: DynamicFixtureState

    var body: some DynamicInstructions {
        Instructions("Nested")
        [
            FixtureTool(name: "tool-a", state: state),
            FixtureTool(name: "tool-b", state: state),
        ] as [any Tool]
    }
}

private final class DynamicFixtureState: @unchecked Sendable {
    enum Selection: Sendable {
        case a
        case b
    }

    struct Storage: Sendable {
        var selection = Selection.a
        var evaluationCount = 0
        var executedTools: [String] = []
    }

    struct Snapshot: Sendable {
        let instructions: String
        let tool: any Tool
    }

    private let storage = Locked(Storage())

    var evaluationCount: Int {
        storage.withLock { $0.evaluationCount }
    }

    var executedTools: [String] {
        storage.withLock { $0.executedTools }
    }

    func select(_ selection: Selection) {
        storage.withLock { $0.selection = selection }
    }

    func snapshotForEvaluation() -> Snapshot {
        storage.withLock { storage in
            storage.evaluationCount += 1
            switch storage.selection {
            case .a:
                return Snapshot(
                    instructions: "Instructions A",
                    tool: FixtureTool(name: "tool-a", state: self)
                )
            case .b:
                return Snapshot(
                    instructions: "Instructions B",
                    tool: FixtureTool(name: "tool-b", state: self)
                )
            }
        }
    }

    func recordExecution(_ name: String) {
        storage.withLock { $0.executedTools.append(name) }
    }
}

private struct FixtureTool: Tool {
    let name: String
    let description = "Records which request-scoped tool instance executed"
    let state: DynamicFixtureState

    typealias Arguments = GeneratedContent

    var parameters: GenerationSchema {
        GeneratedContent.generationSchema
    }

    func call(arguments: GeneratedContent) async throws -> String {
        state.recordExecution(name)
        return name
    }
}

private struct DynamicRequestSnapshot: Sendable, Equatable {
    let instructions: String?
    let tools: [String]
}

private struct DynamicContextModel: LanguageModel {
    typealias UnavailableReason = Never

    let state: DynamicFixtureState
    let continuesAfterTool: Bool
    let snapshots = Locked<[DynamicRequestSnapshot]>([])

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let result = try await run(session: session, type: type)
        return .init(
            content: result.content,
            rawContent: result.raw,
            transcriptEntries: result.entries
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
            Task {
                do {
                    let result = try await run(session: session, type: type)
                    continuation.yield(
                        .init(
                            content: result.content.asPartiallyGenerated(),
                            rawContent: result.raw,
                            transcriptEntries: ArraySlice(result.entries)
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
        return .init(stream: stream)
    }

    private func run<Content: Generable>(
        session: LanguageModelSession,
        type: Content.Type
    ) async throws -> (content: Content, raw: GeneratedContent, entries: ArraySlice<Transcript.Entry>) {
        let first = session.resolvedRequestContext()
        record(first)
        var entries: [Transcript.Entry] = []

        if continuesAfterTool {
            let tool = try #require(first.tools.first)
            state.select(.b)
            let call = Transcript.ToolCall(
                id: "request-a-call",
                toolName: tool.name,
                arguments: GeneratedContent(properties: [:])
            )
            let output = Transcript.ToolOutput(
                id: call.id,
                toolName: call.toolName,
                segments: try await tool.makeOutputSegments(from: call.arguments)
            )
            entries.append(.toolCalls(.init(id: "request-a-calls", [call])))
            entries.append(.toolOutput(output))

            let continuation = session.resolvedRequestContext()
            record(continuation)
        }

        let raw = GeneratedContent("Done")
        return (try Content(raw), raw, ArraySlice(entries))
    }

    private func record(_ context: LanguageModelSession.RequestContext) {
        snapshots.withLock {
            $0.append(
                .init(
                    instructions: context.instructions?.description,
                    tools: context.tools.map(\.name)
                )
            )
        }
    }
}

private enum DynamicFixtureError: Error, Equatable {
    case failed
}

private struct FailingAfterToolModel: LanguageModel {
    typealias UnavailableReason = Never

    let state: DynamicFixtureState
    let snapshots = Locked<[DynamicRequestSnapshot]>([])
    private let didFail = Locked(false)

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let context = session.resolvedRequestContext()
        snapshots.withLock {
            $0.append(
                .init(
                    instructions: context.instructions?.description,
                    tools: context.tools.map(\.name)
                )
            )
        }

        let shouldFail = didFail.withLock { didFail in
            defer { didFail = true }
            return !didFail
        }
        if shouldFail {
            let tool = try #require(context.tools.first)
            _ = try await tool.makeOutputSegments(from: GeneratedContent(properties: [:]))
            state.select(.b)
            throw DynamicFixtureError.failed
        }

        let raw = GeneratedContent("Done")
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
            $0.finish(throwing: DynamicFixtureError.failed)
        }
        return .init(stream: stream)
    }
}

private final class DynamicCancellationControl: @unchecked Sendable {
    let started: AsyncStream<Void>

    private let startedContinuation: AsyncStream<Void>.Continuation
    private let cancellationContinuation = Locked<CheckedContinuation<Void, any Error>?>(nil)

    init() {
        (started, startedContinuation) = AsyncStream.makeStream()
    }

    func signalStarted() {
        startedContinuation.yield(())
    }

    func waitForCancellation() async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let isAlreadyCancelled = cancellationContinuation.withLock { stored in
                    guard !Task.isCancelled else { return true }
                    stored = continuation
                    return false
                }
                if isAlreadyCancelled {
                    continuation.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            let continuation = cancellationContinuation.withLock { stored in
                defer { stored = nil }
                return stored
            }
            continuation?.resume(throwing: CancellationError())
        }
    }
}

private struct CancellingAfterToolModel: LanguageModel {
    typealias UnavailableReason = Never

    let state: DynamicFixtureState
    let control: DynamicCancellationControl
    let snapshots = Locked<[DynamicRequestSnapshot]>([])
    private let didSuspend = Locked(false)

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let context = session.resolvedRequestContext()
        snapshots.withLock {
            $0.append(
                .init(
                    instructions: context.instructions?.description,
                    tools: context.tools.map(\.name)
                )
            )
        }

        let shouldSuspend = didSuspend.withLock { didSuspend in
            defer { didSuspend = true }
            return !didSuspend
        }
        if shouldSuspend {
            let tool = try #require(context.tools.first)
            _ = try await tool.makeOutputSegments(from: GeneratedContent(properties: [:]))
            state.select(.b)
            control.signalStarted()
            try await control.waitForCancellation()
        }

        let raw = GeneratedContent("Done")
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
            $0.finish(throwing: CancellationError())
        }
        return .init(stream: stream)
    }
}
