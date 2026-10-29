/// A decision about how to handle a tool call.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public enum ToolExecutionDecision: Sendable {
    /// Execute the tool call using the associated tool.
    case execute

    /// Stop the session after tool calls are generated without executing them.
    case stop

    /// Provide tool output without executing the tool.
    ///
    /// Use this to supply results from an external system or cached responses.
    case provideOutput([Transcript.Segment])
}

/// A delegate that observes and controls tool execution for a session.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public protocol ToolExecutionDelegate: Sendable {
    /// Notifies the delegate when the model generates tool calls.
    ///
    /// - Parameters:
    ///   - toolCalls: The tool calls produced by the model.
    ///   - session: The session that generated the tool calls.
    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async

    /// Asks the delegate how to handle a tool call.
    ///
    /// Return `.execute` to run the tool, `.stop` to halt after tool calls are generated,
    /// or `.provideOutput` to supply output without executing the tool.
    /// - Parameters:
    ///   - toolCall: The tool call to evaluate.
    ///   - session: The session requesting the decision.
    func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
        -> ToolExecutionDecision

    /// Notifies the delegate after a tool call produces output.
    ///
    /// - Parameters:
    ///   - toolCall: The tool call that was handled.
    ///   - output: The output sent back to the model.
    ///   - session: The session that executed the tool call.
    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async

    /// Notifies the delegate when a tool call fails.
    ///
    /// - Parameters:
    ///   - toolCall: The tool call that failed.
    ///   - error: The underlying error raised during execution.
    ///   - session: The session that attempted the tool call.
    func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: LanguageModelSession
    ) async

    /// Asks the delegate for messages to add to a response in progress,
    /// after a round of tool calls and before the model's next request.
    ///
    /// Use this to let someone steer a long response while it runs:
    /// return what they've sent since the last round,
    /// and the model reads it after the round's tool outputs.
    /// Each message joins the response's transcript entries as a prompt,
    /// in order, and is sent in the request in the same form,
    /// so a later response replays it exactly as it was sent.
    ///
    /// The session asks after every round of tool calls
    /// in providers whose tool loop continues after a round.
    /// Messages the delegate is still holding when a response ends
    /// without another round stay with the app,
    /// which can send them as the next prompt.
    /// Steering doesn't reset ``GenerationOptions/maximumToolRounds``.
    ///
    /// Once returned, the messages are the response's.
    /// If the response then fails,
    /// they're handled like the rest of its entries,
    /// as the session's ``LanguageModelSession/transcriptErrorHandlingPolicy`` says:
    /// a streaming response's checkpoint may keep them,
    /// and a reverted or nonstreaming response doesn't.
    /// They aren't handed back, so resend any that matter.
    ///
    /// ``SystemLanguageModel`` doesn't ask:
    /// Foundation Models runs its own tool loop.
    ///
    /// - Parameter session: The session generating the response.
    /// - Returns: The messages to add, oldest first; empty for none.
    func pendingMessages(in session: LanguageModelSession) async -> [Prompt]
}

// MARK: - Default Implementations

extension ToolExecutionDelegate {
    /// Provides a default no-op implementation.
    public func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {}

    /// Provides a default decision that executes the tool call.
    public func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        .execute
    }

    /// Provides a default no-op implementation.
    public func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {}

    /// Provides a default implementation that adds no messages.
    public func pendingMessages(in session: LanguageModelSession) async -> [Prompt] { [] }

    /// Provides a default no-op implementation.
    public func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: LanguageModelSession
    ) async {}
}
