import Foundation
import Testing

@testable import AnyLanguageModel

/// Waits, then answers: a stand-in for a tool that calls a service.
private struct WaitingTool: Tool {
    let name: String
    let description = "Waits, then answers"
    let milliseconds: Int

    init(name: String = "wait", milliseconds: Int) {
        self.name = name
        self.milliseconds = milliseconds
    }

    @Generable
    struct Arguments {
        @Guide(description: "The answer")
        var answer: String
    }

    func call(arguments: Arguments) async throws -> String {
        try await Task.sleep(for: .milliseconds(milliseconds))
        return arguments.answer
    }
}

private struct FailingTool: Tool {
    let name = "fail"
    let description = "Fails"

    @Generable
    struct Arguments {
        @Guide(description: "Ignored")
        var message: String
    }

    func call(arguments: Arguments) async throws -> String {
        try await Task.sleep(for: .milliseconds(20))
        throw FailingToolError.failed
    }
}

private enum FailingToolError: Error { case failed }

/// Records what the delegate hears; decides each call with `decide`.
private actor RoundSpy: ToolExecutionDelegate {
    private(set) var executed: [String] = []
    private(set) var failed: [String] = []
    private let decide: @Sendable (Transcript.ToolCall) -> ToolExecutionDecision

    init(decide: @escaping @Sendable (Transcript.ToolCall) -> ToolExecutionDecision = { _ in .execute }) {
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
        executed.append(toolCall.id)
    }

    func didFailToolCall(_ toolCall: Transcript.ToolCall, error: any Error, in session: LanguageModelSession) async {
        failed.append(toolCall.id)
    }
}

private func call(_ id: String, _ tool: String, answer: String = "ok") throws -> Transcript.ToolCall {
    Transcript.ToolCall(
        id: id,
        toolName: tool,
        arguments: try GeneratedContent(json: #"{"answer":"\#(answer)","message":"x"}"#)
    )
}

private func outputs(_ outcome: ToolResolutionOutcome) -> [(id: String, text: String)] {
    guard case .invocations(let results) = outcome else { return [] }
    return results.map { result in
        let text = result.output.segments.compactMap { segment -> String? in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return try? structure.content.value(String.self)
            default: return nil
            }
        }.joined()
        return (result.call.id, text)
    }
}

@Suite("Tool rounds")
struct ToolRoundTests {
    @Test func callsInARoundRunAtOnce() async throws {
        let slow = WaitingTool(name: "slow", milliseconds: 300)
        let fast = WaitingTool(name: "fast", milliseconds: 300)
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [slow, fast])
        let calls = [try call("a", "slow"), try call("b", "fast")]

        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            _ = try await resolveToolRound(calls, tools: [slow, fast], session: session)
        }

        // One after another would take 600 ms.
        #expect(elapsed < .milliseconds(500))
    }

    @Test func resultsKeepTheModelsOrder() async throws {
        let slow = WaitingTool(name: "slow", milliseconds: 200)
        let fast = WaitingTool(name: "fast", milliseconds: 10)
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [slow, fast])
        let calls = [try call("first", "slow", answer: "one"), try call("second", "fast", answer: "two")]

        let outcome = try await resolveToolRound(calls, tools: [slow, fast], session: session)

        #expect(outputs(outcome).map(\.id) == ["first", "second"])
        #expect(outputs(outcome).map(\.text) == ["one", "two"])
    }

    @Test func theFirstErrorEndsTheRoundAndCancelsTheOthers() async throws {
        let slow = WaitingTool(name: "slow", milliseconds: 2_000)
        let failing = FailingTool()
        let spy = RoundSpy()
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [slow, failing])
        session.toolExecutionDelegate = spy
        let calls = [try call("slow", "slow"), try call("fail", "fail")]

        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            await #expect(throws: LanguageModelSession.ToolCallError.self) {
                _ = try await resolveToolRound(calls, tools: [slow, failing], session: session)
            }
        }

        // The slow call was cancelled rather than waited for, and only the call that failed
        // is reported.
        #expect(elapsed < .milliseconds(1_000))
        #expect(await spy.failed == ["fail"])
        #expect(await spy.executed.isEmpty)
    }

    @Test func aStopDecisionEndsTheRoundBeforeAnyCallRuns() async throws {
        let tool = WaitingTool(milliseconds: 10)
        let spy = RoundSpy { $0.id == "second" ? .stop : .execute }
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [tool])
        session.toolExecutionDelegate = spy
        let calls = [try call("first", "wait"), try call("second", "wait")]

        let outcome = try await resolveToolRound(calls, tools: [tool], session: session)

        guard case .stop(let stopped) = outcome else {
            Issue.record("Expected the round to stop")
            return
        }
        #expect(stopped.map(\.id) == ["first", "second"])
        #expect(await spy.executed.isEmpty)
    }

    @Test func providedOutputAndMissingToolsAnswerWithoutRunning() async throws {
        let tool = WaitingTool(milliseconds: 10)
        let spy = RoundSpy { $0.id == "provided" ? .provideOutput([.text(.init(content: "cached"))]) : .execute }
        let session = LanguageModelSession(model: MockLanguageModel(), tools: [tool])
        session.toolExecutionDelegate = spy
        let calls = [
            try call("provided", "wait"), try call("missing", "nowhere"), try call("run", "wait", answer: "ran"),
        ]

        let outcome = try await resolveToolRound(calls, tools: [tool], session: session)

        #expect(outputs(outcome).map(\.text) == ["cached", "Tool not found: nowhere", "ran"])
        #expect(Set(await spy.executed) == ["provided", "missing", "run"])
    }
}
