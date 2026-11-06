import Foundation
import Testing

@testable import AnyLanguageModel

/// What happened, in order, across tools and the delegate.
private actor EventLog {
    private(set) var events: [String] = []

    func record(_ event: String) {
        events.append(event)
    }
}

/// Opens once; everyone waiting goes on.
private actor Gate {
    private var isOpen = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    func open() {
        isOpen = true
        for continuation in waiting { continuation.resume() }
        waiting = []
    }

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiting.append($0) }
    }
}

/// Waits for the gate, then answers. With a gate no one opens, it waits until it's cancelled.
private struct GatedTool: Tool {
    let name: String
    let description = "Waits for a gate, then answers"
    let gate: Gate
    var opens: Gate?
    var log: EventLog?

    @Generable
    struct Arguments {
        @Guide(description: "The answer")
        var answer: String
    }

    func call(arguments: Arguments) async throws -> String {
        await opens?.open()
        await withTaskCancellationHandler {
            await gate.wait()
        } onCancel: {
            Task { await gate.open() }
        }
        if Task.isCancelled {
            await log?.record("\(name) cancelled")
            throw CancellationError()
        }
        await log?.record("\(name) finished")
        return arguments.answer
    }
}

/// Answers at once.
private struct AnsweringTool: Tool {
    let name: String
    let description = "Answers"

    @Generable
    struct Arguments {
        @Guide(description: "The answer")
        var answer: String
    }

    func call(arguments: Arguments) async throws -> String {
        arguments.answer
    }
}

/// Answers, then opens a gate.
private struct OpeningTool: Tool {
    let name: String
    let description = "Answers, then opens a gate"
    let opens: Gate
    let log: EventLog

    @Generable
    struct Arguments {
        @Guide(description: "The answer")
        var answer: String
    }

    func call(arguments: Arguments) async throws -> String {
        await log.record("\(name) finished")
        await opens.open()
        return arguments.answer
    }
}

/// Throws, whether or not the round has been cancelled.
private struct FailingTool: Tool {
    let name: String
    let description = "Fails"
    var opens: Gate?
    var after: Gate?

    @Generable
    struct Arguments {
        @Guide(description: "Ignored")
        var answer: String
    }

    func call(arguments: Arguments) async throws -> String {
        await opens?.open()
        await after?.wait()
        throw FailingToolError.failed
    }
}

private enum FailingToolError: Error { case failed }

/// Records what the delegate hears, and takes a moment over each, so overlapping callbacks
/// would show; decides each call with `decide`.
private final class RoundSpy: ToolExecutionDelegate, @unchecked Sendable {
    let log: EventLog
    private let decide: @Sendable (Transcript.ToolCall) -> ToolExecutionDecision

    init(
        log: EventLog = EventLog(),
        decide: @escaping @Sendable (Transcript.ToolCall) -> ToolExecutionDecision = { _ in .execute }
    ) {
        self.log = log
        self.decide = decide
    }

    func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
        -> ToolExecutionDecision
    {
        decide(toolCall)
    }

    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {
        await log.record("begin \(toolCall.id)")
        try? await Task.sleep(for: .milliseconds(20))
        await log.record("end \(toolCall.id)")
    }

    func didFailToolCall(_ toolCall: Transcript.ToolCall, error: any Error, in session: LanguageModelSession) async {
        await log.record("failed \(toolCall.id)")
        try? await Task.sleep(for: .milliseconds(100))
    }
}

/// Records the tool rounds in `history` once `waitsFor` opens.
private struct HistoryReadingTool: Tool {
    let name: String
    let description = "Reads the history"
    var waitsFor: Gate?
    var opens: Gate?
    var seen: Locked<[String]>?

    @LanguageModelSession.SessionProperty(\.history) private var history

    @Generable
    struct Arguments {
        @Guide(description: "The answer")
        var answer: String
    }

    func call(arguments: Arguments) async throws -> String {
        await waitsFor?.wait()
        let entries = history.compactMap { entry -> String? in
            switch entry {
            case .toolCalls(let calls): return "calls " + calls.map(\.id).joined(separator: " ")
            case .toolOutput(let output): return "output " + output.id
            default: return nil
            }
        }
        seen?.withLock { $0 = entries }
        await opens?.open()
        return arguments.answer
    }
}

private func call(_ id: String, _ tool: String, answer: String = "ok") throws -> Transcript.ToolCall {
    Transcript.ToolCall(
        id: id,
        toolName: tool,
        arguments: try GeneratedContent(json: #"{"answer":"\#(answer)"}"#)
    )
}

private func resolve(
    _ calls: [Transcript.ToolCall],
    in session: LanguageModelSession
) async throws -> ProfileToolResolution {
    try await session.resolveProfileToolCalls(
        calls,
        requestContext: session.resolvedRequestContext(),
        currentRoundEntries: [.toolCalls(Transcript.ToolCalls(calls))]
    )
}

private func outputs(_ resolution: ProfileToolResolution) -> [(id: String, text: String)] {
    guard case .invocations(let invocations) = resolution else { return [] }
    return invocations.map { invocation in
        let text = invocation.output.segments.compactMap { segment -> String? in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return try? structure.content.value(String.self)
            default: return nil
            }
        }.joined()
        return (invocation.call.id, text)
    }
}

@Suite("Tool rounds")
struct ToolRoundTests {
    /// Each call waits for the other to start, so the round can only finish if they run at once.
    @Test(.timeLimit(.minutes(1)))
    func callsInARoundRunAtOnce() async throws {
        let aStarted = Gate()
        let bStarted = Gate()
        let a = GatedTool(name: "a", gate: bStarted, opens: aStarted)
        let b = GatedTool(name: "b", gate: aStarted, opens: bStarted)
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [a, b])

        let resolution = try await resolve(
            [try call("1", "a", answer: "one"), try call("2", "b", answer: "two")],
            in: session
        )

        #expect(outputs(resolution).map(\.text) == ["one", "two"])
    }

    /// The second call reads the history after the first has finished,
    /// and sees the earlier round and this round's calls, but not the first call's output.
    @Test(.timeLimit(.minutes(1)))
    func aToolSeesEarlierRoundsAndCurrentCallsButNotOtherCallsOutputs() async throws {
        let firstFinished = Gate()
        let seen = Locked<[String]>([])
        let session = LanguageModelSession(
            model: MockLanguageModel(),
            tools: [
                HistoryReadingTool(name: "first", opens: firstFinished),
                HistoryReadingTool(name: "second", waitsFor: firstFinished, seen: seen),
            ]
        )
        let earlierCall = try call("0", "first")
        let earlierRound: [Transcript.Entry] = [
            .toolCalls(Transcript.ToolCalls([earlierCall])),
            .toolOutput(Transcript.ToolOutput(id: earlierCall.id, toolName: earlierCall.toolName, segments: [])),
        ]
        let calls = [try call("1", "first"), try call("2", "second")]

        _ = try await session.resolveProfileToolCalls(
            calls,
            requestContext: session.resolvedRequestContext(),
            currentRoundEntries: earlierRound + [.toolCalls(Transcript.ToolCalls(calls))]
        )

        #expect(seen.withLock { $0 } == ["calls 0", "output 0", "calls 1 2"])
    }

    /// The first call finishes last, yet the results and the delegate keep the model's order,
    /// and the delegate hears about one call at a time.
    @Test(.timeLimit(.minutes(1)))
    func theDelegateHearsInTheModelsOrderOneAtATime() async throws {
        let log = EventLog()
        let secondFinished = Gate()
        let first = GatedTool(name: "first", gate: secondFinished, log: log)
        let second = OpeningTool(name: "second", opens: secondFinished, log: log)
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [first, second])
        session.toolExecutionDelegate = RoundSpy(log: log)

        let resolution = try await resolve(
            [try call("1", "first", answer: "one"), try call("2", "second", answer: "two")],
            in: session
        )

        #expect(outputs(resolution).map(\.id) == ["1", "2"])
        #expect(outputs(resolution).map(\.text) == ["one", "two"])
        #expect(
            await log.events == ["second finished", "first finished", "begin 1", "end 1", "begin 2", "end 2"]
        )
    }

    /// When a call throws, the others are cancelled before the delegate hears about the error.
    @Test(.timeLimit(.minutes(1)))
    func theFirstErrorCancelsTheOthersBeforeTheDelegateHears() async throws {
        let log = EventLog()
        let waiting = GatedTool(name: "waiting", gate: Gate(), log: log)
        let spy = RoundSpy(log: log)
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [waiting, FailingTool(name: "fail")])
        session.toolExecutionDelegate = spy

        await #expect(throws: LanguageModelSession.ToolCallError.self) {
            _ = try await resolve([try call("1", "waiting"), try call("2", "fail")], in: session)
        }

        #expect(await log.events == ["waiting cancelled", "failed 2"])
    }

    /// Two calls fail, the second as the first does, while the delegate is slow to hear about
    /// the first; only the error the round throws reaches it.
    @Test(.timeLimit(.minutes(1)))
    func onlyTheRoundsErrorReachesTheDelegate() async throws {
        let log = EventLog()
        let firstFailing = Gate()
        let session = LanguageModelSession(
            model: MockLanguageModel(),
            tools: [FailingTool(name: "fail", opens: firstFailing), FailingTool(name: "fail too", after: firstFailing)]
        )
        session.toolExecutionDelegate = RoundSpy(log: log)

        let error = await #expect(throws: LanguageModelSession.ToolCallError.self) {
            _ = try await resolve([try call("1", "fail"), try call("2", "fail too")], in: session)
        }

        let reported = try #require(error.map { $0.tool.name == "fail" ? "failed 1" : "failed 2" })
        #expect(await log.events == [reported])
    }

    @Test func aStopDecisionEndsTheRoundBeforeAnyCallRuns() async throws {
        let log = EventLog()
        let spy = RoundSpy(log: log) { $0.id == "second" ? .stop : .execute }
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [AnsweringTool(name: "answer")])
        session.toolExecutionDelegate = spy

        let resolution = try await resolve([try call("first", "answer"), try call("second", "answer")], in: session)

        guard case .stop(let stopped) = resolution else {
            Issue.record("Expected the round to stop")
            return
        }
        #expect(stopped.map(\.id) == ["first", "second"])
        #expect(await log.events.isEmpty)
    }

    @Test func providedOutputAndMissingToolsAnswerWithoutRunning() async throws {
        let log = EventLog()
        let spy = RoundSpy(log: log) {
            $0.id == "provided" ? .provideOutput([.text(.init(content: "cached"))]) : .execute
        }
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [AnsweringTool(name: "answer")])
        session.toolExecutionDelegate = spy
        let calls = [
            try call("provided", "answer"), try call("missing", "nowhere"), try call("run", "answer", answer: "ran"),
        ]

        let resolution = try await resolve(calls, in: session)

        #expect(outputs(resolution).map(\.text) == ["cached", "Tool not found: nowhere", "ran"])
        #expect(
            await log.events == [
                "begin provided", "end provided", "begin missing", "end missing", "begin run", "end run",
            ]
        )
    }
}
