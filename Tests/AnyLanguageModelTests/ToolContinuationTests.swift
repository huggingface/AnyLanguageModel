import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

@Suite("Tool continuation history")
struct ToolContinuationTests {
    @Test func explicitEmptyOfferedToolsDoesNotExecuteASessionTool() async throws {
        let calls = Locked(0)
        let tool = OfferedToolProbe(calls: calls)
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [tool])
        let call = Transcript.ToolCall(id: "call_0", toolName: tool.name, arguments: .init("input"))
        let result = try await session.resolveProfileToolCalls(
            [call],
            requestContext: session.resolvedRequestContext(),
            currentRoundEntries: [.toolCalls(.init([call]))],
            offeredTools: []
        )
        guard case .invocations(let invocations) = result else {
            Issue.record("Expected the existing missing-tool outcome")
            return
        }
        #expect(calls.withLock { $0 } == 0)
        #expect(invocations.count == 1)
    }

    @Test func removingDurableParentDoesNotCommitItsPendingOutput() async throws {
        let round = Self.round("pending")
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("History") }
                .model(MockLanguageModel())
                .historyTransform { entries in
                    let properties = SessionPropertyBinding.values!
                    properties.history = []
                    properties.history = properties.history
                    return entries
                },
            history: [round[0]]
        )
        _ = try await session.resolvedRequestContext(including: round)
        #expect(session.properties.history.isEmpty)
    }

    @Test func ordinaryContinuationKeepsTranscriptAndOpaqueMessages() {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let messages = toolContinuationMessages(
            from: Transcript(entries: previous),
            replacing: current,
            with: current,
            hasDynamicProfile: false,
            convert: Array.init
        )
        #expect(messages == previous + current)
    }

    @Test func profileContinuationReplacesOnlyTheUnchangedCurrentRound() {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let opaque: [Transcript.Entry] = [.response(.init(assetIDs: [], segments: []))]
        let messages = toolContinuationMessages(
            from: Transcript(entries: previous + current),
            replacing: current,
            with: opaque,
            hasDynamicProfile: true,
            convert: Array.init
        )
        #expect(messages == previous + opaque)
        let projection = previous + current.reversed()
        let reordered = toolContinuationMessages(
            from: Transcript(entries: projection),
            replacing: current,
            with: opaque,
            hasDynamicProfile: true,
            convert: Array.init
        )
        #expect(reordered == projection)
    }

    @Test(arguments: [false, true])
    func reusedCallIDDoesNotDiscardANewTurnOutput(profile: Bool) async throws {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let session: LanguageModelSession
        if profile {
            session = LanguageModelSession(
                profile: LanguageModelSession.Profile { Instructions("History") }
                    .model(MockLanguageModel())
                    .historyTransform { entries in
                        let properties = SessionPropertyBinding.values!
                        properties.history = properties.history
                        return entries
                    },
                history: previous
            )
        } else {
            session = LanguageModelSession(model: MockLanguageModel(), transcript: Transcript(entries: previous))
        }
        let context = try await session.resolvedRequestContext(including: current)
        #expect(Self.outputs(context.transcript) == Self.outputs(Transcript(entries: previous + current)))
        #expect(Array(session.properties.history) == previous)
    }

    private static func round(_ result: String) -> [Transcript.Entry] {
        let call = Transcript.ToolCall(id: "call_0", toolName: "fixture", arguments: .init(properties: [:]))
        return [
            .toolCalls(.init([call])),
            .toolOutput(.init(id: call.id, toolName: call.toolName, segments: [.text(.init(content: result))])),
        ]
    }

    private static func outputs(_ transcript: Transcript) -> [Transcript.ToolOutput] {
        transcript.compactMap { entry in
            if case .toolOutput(let output) = entry { return output }
            return nil
        }
    }
}

private struct OfferedToolProbe: Tool {
    let name = "offeredProbe"
    let description = "Record unexpected executions"
    let calls: Locked<Int>

    func call(arguments: String) async throws -> String {
        calls.withLock { $0 += 1 }
        return "executed"
    }
}
