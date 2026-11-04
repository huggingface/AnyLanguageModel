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

    // Without its parent, the pending output's round-scoped key
    // is the same as the durable output's.
    @Test(arguments: [false, true])
    func removingPendingParentDoesNotCommitItsOutputWithAReusedCallID(sameResult: Bool) async throws {
        let previous = Self.round("previous")
        let current = Self.round(sameResult ? "previous" : "current")
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("History") }
                .model(MockLanguageModel())
                .historyTransform { entries in
                    let properties = SessionPropertyBinding.values!
                    properties.history.removeAll { $0.id == current[0].id }
                    return entries
                },
            history: previous
        )
        _ = try await session.resolvedRequestContext(including: current)
        #expect(Array(session.properties.history) == previous)
    }

    // Removing the durable output and the pending parent
    // gives the pending output the durable output's key.
    @Test(arguments: [false, true])
    func removingDurableOutputAndPendingParentDoesNotCommitThePendingOutput(edited: Bool) async throws {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let kept = edited ? Self.round("edited")[1] : current[1]
        let visible = Locked<[Transcript.Entry]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("History") }
                .model(MockLanguageModel())
                .historyTransform { entries in
                    let properties = SessionPropertyBinding.values!
                    properties.history = Transcript.HistoryView([previous[0], kept])
                    visible.withLock { $0 = Array(properties.history) }
                    return entries
                },
            history: previous
        )
        _ = try await session.resolvedRequestContext(including: current)
        let durable = edited ? [previous[0], kept] : [previous[0]]
        #expect(visible.withLock { $0 } == durable + current)
        #expect(Array(session.properties.history) == durable)
    }

    // A round that a write adds owns its outputs, even with a pending output's call ID,
    // but an unchanged copy of the pending output in it stays pending.
    @Test(arguments: [false, true])
    func newRoundKeepsItsOutputWithAPendingCallID(copiesPendingOutput: Bool) async throws {
        let current = Self.round("current")
        let added = Self.round("added")
        let addedRound = [added[0], copiesPendingOutput ? current[1] : added[1]]
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile { Instructions("History") }
                .model(MockLanguageModel())
                .historyTransform { entries in
                    let properties = SessionPropertyBinding.values!
                    properties.history.append(contentsOf: addedRound)
                    return entries
                }
        )
        _ = try await session.resolvedRequestContext(including: current)
        #expect(Array(session.properties.history) == (copiesPendingOutput ? [added[0]] : added))
    }

    @Test(arguments: [false, true])
    func rollbackRemovesAnUnchangedOutputThatTookADurableOutputsKey(edited: Bool) {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let history = previous + current
        let appended = AppendedTranscriptEntries(
            keys: Set(keyedTranscriptEntries(history).suffix(current.count).map(\.key)),
            history: history
        )
        let kept = edited ? Self.round("edited")[1] : current[1]
        let remaining = removingAppendedTranscriptEntries(appended, from: [previous[0], kept])
        #expect(remaining == (edited ? [previous[0], kept] : [previous[0]]))
    }

    // A callback that inserts a round before an appended output gives the output that round's key.
    // An unchanged appended output stays appended; an edited one belongs to the new round.
    @Test(arguments: [false, true])
    func rollbackRemovesAnUnchangedOutputThatACallbackMovedIntoANewRound(edited: Bool) {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let history = previous + current
        let appended = AppendedTranscriptEntries(
            keys: Set(keyedTranscriptEntries(history).suffix(current.count).map(\.key)),
            history: history
        )
        let inserted = Self.round("inserted")[0]
        let kept = edited ? Self.round("edited")[1] : current[1]
        let remaining = removingAppendedTranscriptEntries(appended, from: previous + [current[0], inserted, kept])
        #expect(remaining == previous + (edited ? [inserted, kept] : [inserted]))
    }

    // A later round's output that equals an appended output
    // belongs to the later response while the appended output keeps its key.
    @Test func rollbackKeepsALaterOutputThatEqualsAnAppendedOutput() {
        let previous = Self.round("previous")
        let current = Self.round("current")
        let history = previous + current
        let appended = AppendedTranscriptEntries(
            keys: Set(keyedTranscriptEntries(history).suffix(current.count).map(\.key)),
            history: history
        )
        let later = [Self.round("later")[0], current[1]]
        let remaining = removingAppendedTranscriptEntries(appended, from: history + later)
        #expect(remaining == previous + later)
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
