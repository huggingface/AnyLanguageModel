import Foundation

/// Bounds the tool rounds in one response, as its options say.
///
/// A model that keeps calling tools would otherwise run tools
/// and send requests until the caller cancels.
/// A response allows at most ``GenerationOptions/maximumToolRounds`` rounds of tool calls, 8 by default.
/// A round that repeats the previous one runs like any other:
/// calling the same tool with the same arguments again can be what's asked,
/// and the limit is what ends a loop.
final class ToolRoundLimit: @unchecked Sendable {
    /// The provider name used in error messages.
    let provider: String
    /// The most rounds the response may take.
    let maximumRounds: Int

    private let roundCount = Locked(0)

    init(provider: String, options: GenerationOptions) {
        self.provider = provider
        self.maximumRounds = options.maximumToolRounds
    }

    /// Records a round of tool calls before any of them runs.
    ///
    /// - Parameter completed: The response's entries so far: the rounds that ran, each call
    ///   followed by its output.
    /// - Throws: ``LanguageModelSession/ToolRoundLimitExceeded`` with `completed`
    ///   when the round is past the most rounds the options allow.
    func record(completed: [Transcript.Entry]) throws {
        try roundCount.withLock { rounds in
            rounds += 1
            if rounds > maximumRounds {
                throw LanguageModelSession.ToolRoundLimitExceeded(
                    limit: maximumRounds,
                    provider: provider,
                    transcriptEntries: ArraySlice(completed)
                )
            }
        }
    }
}

extension LanguageModelSession {
    /// A response that called tools in more rounds than ``GenerationOptions/maximumToolRounds`` allows.
    ///
    /// The rounds that ran are in `transcriptEntries`, each tool call followed by its output,
    /// so a caller can add them to the transcript and continue from there.
    /// The round past the limit didn't run, and isn't included.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel.
    public struct ToolRoundLimitExceeded: Error, CustomStringConvertible, Sendable {
        /// The most rounds the response could take.
        public let limit: Int

        /// The provider whose response called tools past the limit.
        public let provider: String

        /// The rounds that ran, each tool call followed by its output.
        public let transcriptEntries: ArraySlice<Transcript.Entry>

        public var description: String {
            "Exceeded maximum tool iterations (\(limit)) while processing \(provider) tool calls."
        }
    }
}
