/// A tool call's result in a round: the call, and the output sent back to the model.
struct ToolInvocationResult: Sendable {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

/// How a round of tool calls ended.
enum ToolResolutionOutcome: Sendable {
    /// The delegate stopped the session after the calls were generated.
    case stop(calls: [Transcript.ToolCall])
    /// The calls' results, in the order the model made the calls.
    case invocations([ToolInvocationResult])
}

/// Resolves a round of tool calls the same way for every provider whose tool loop runs here.
///
/// The delegate hears about the round, then decides each call in order; a `.stop` ends the
/// round before any call runs. The calls then run at once, as Foundation Models runs them —
/// a tool can be called concurrently with itself or with other tools, so one that isn't safe
/// to run alongside others serializes its own work. Results keep the order the model gave
/// the calls in. When a call throws, the others are cancelled and the first error is
/// reported, as a throwing task group does.
func resolveToolRound(
    _ calls: [Transcript.ToolCall],
    tools: [any Tool],
    session: LanguageModelSession
) async throws -> ToolResolutionOutcome {
    let delegate = session.toolExecutionDelegate
    await delegate?.didGenerateToolCalls(calls, in: session)
    guard !calls.isEmpty else { return .invocations([]) }

    var decisions: [ToolExecutionDecision] = []
    decisions.reserveCapacity(calls.count)
    for call in calls {
        let decision = await delegate?.toolCallDecision(for: call, in: session) ?? .execute
        if case .stop = decision { return .stop(calls: calls) }
        decisions.append(decision)
    }

    var toolsByName: [String: any Tool] = [:]
    for tool in tools where toolsByName[tool.name] == nil {
        toolsByName[tool.name] = tool
    }

    return try await withThrowingTaskGroup(of: (Int, ToolInvocationResult).self) { group in
        for (index, call) in calls.enumerated() {
            let decision = decisions[index]
            let tool = toolsByName[call.toolName]
            group.addTask {
                (index, try await invokeToolCall(call, decision: decision, tool: tool, session: session))
            }
        }
        var results = [ToolInvocationResult?](repeating: nil, count: calls.count)
        for try await (index, result) in group {
            results[index] = result
        }
        return .invocations(results.compactMap { $0 })
    }
}

/// One call of a round: its output, from the tool or from the delegate's decision.
private func invokeToolCall(
    _ call: Transcript.ToolCall,
    decision: ToolExecutionDecision,
    tool: (any Tool)?,
    session: LanguageModelSession
) async throws -> ToolInvocationResult {
    let delegate = session.toolExecutionDelegate
    let segments: [Transcript.Segment]
    switch decision {
    case .provideOutput(let provided):
        segments = provided
    case .stop, .execute:
        // A `.stop` ends the round before any call runs, so here it's `.execute`.
        guard let tool else {
            segments = [.text(.init(content: "Tool not found: \(call.toolName)"))]
            break
        }
        do {
            segments = try await tool.makeOutputSegments(from: call.arguments)
        } catch {
            // Cancelled because another call failed: that one's error is the round's.
            if !Task.isCancelled {
                await delegate?.didFailToolCall(call, error: error, in: session)
            }
            throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
        }
    }
    let output = Transcript.ToolOutput(id: call.id, toolName: tool?.name ?? call.toolName, segments: segments)
    await delegate?.didExecuteToolCall(call, output: output, in: session)
    return ToolInvocationResult(call: call, output: output)
}
