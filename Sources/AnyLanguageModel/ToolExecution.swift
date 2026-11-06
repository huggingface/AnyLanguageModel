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

extension LanguageModelSession {
    /// A response that a ``ToolExecutionDelegate`` stopped
    /// before the model generated a value of the response's type.
    ///
    /// When a delegate returns ``ToolExecutionDecision/stop``,
    /// the response ends with the tool calls that stopped it.
    /// `respond(to:)` returns an empty value,
    /// even if the model wrote text before the tool calls:
    /// `""` for `String`, and otherwise the first of `{}`, `null`, and `[]`
    /// that decodes as the type.
    /// After a profile hands the response off to another model,
    /// a `String` response starts with the text from before the hand-off.
    /// A stream keeps its last snapshot if that snapshot decodes as the complete type,
    /// so a `String` stream ends with the text so far.
    /// Otherwise, a stream also ends with an empty value.
    /// Structured streams from MLX and llama.cpp follow the rule for `respond(to:)`.
    /// A type with required properties can't be empty,
    /// and zero or `false` would look like the model's answer,
    /// so a response of such a type, a number, or a Boolean
    /// throws this error instead.
    ///
    /// The rounds that ran are in `transcriptEntries`,
    /// each tool call followed by its output,
    /// and then the tool calls that the delegate stopped, without output.
    /// As with any error, the session keeps only the prompt,
    /// so a caller can add these entries to the transcript and continue from there.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel.
    public struct ToolExecutionStopped: Error, CustomStringConvertible, Sendable {
        /// The provider whose response a delegate stopped.
        public let provider: String

        /// The rounds that ran, each tool call followed by its output,
        /// and then the tool calls that the delegate stopped.
        public let transcriptEntries: ArraySlice<Transcript.Entry>

        /// The usage of a stream's requests, which the session counts when the stream throws this,
        /// because the stream can end without a snapshot that reports them.
        internal var requests = RequestsUsage()

        public var description: String {
            "A tool execution delegate stopped the \(provider) response "
                + "before the model generated a value of the response type."
        }

        /// This error for a response that a model continued after a handoff,
        /// with the entries and usage of the rounds before the handoff.
        func continuing(after entries: [Transcript.Entry], usage: LanguageModelSession.Usage) -> Self {
            var earlier = usage
            earlier.add(requests.earlier)
            return Self(
                provider: provider,
                transcriptEntries: ArraySlice(entries + transcriptEntries),
                requests: RequestsUsage(earlier: earlier, final: requests.final)
            )
        }
    }
}
