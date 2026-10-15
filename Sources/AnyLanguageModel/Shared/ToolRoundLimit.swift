import Foundation

/// Bounds the tool rounds in one response, as its options say.
///
/// A model that keeps calling tools would otherwise run tools
/// and send requests until the caller cancels.
/// A response allows at most ``GenerationOptions/maximumToolRounds`` rounds of tool calls, 8 by default.
/// A round that repeats the previous one runs like any other:
/// calling the same tool with the same arguments again can be what's asked,
/// and the limit is what ends a loop.
struct ToolRoundLimit {
    /// The provider name used in error messages.
    let provider: String
    /// The most rounds the response may take.
    let maximumRounds: Int

    private var taken = 0

    init(provider: String, options: GenerationOptions) {
        self.provider = provider
        self.maximumRounds = options.maximumToolRounds
    }

    /// Records a round of tool calls before any of them runs.
    ///
    /// - Throws: A decoding failure when the round is past the most rounds the options allow.
    mutating func record() throws {
        taken += 1
        if taken > maximumRounds {
            throw LanguageModelSession.GenerationError.decodingFailure(
                .init(
                    debugDescription:
                        "Exceeded maximum tool iterations (\(maximumRounds)) while processing \(provider) tool calls."
                )
            )
        }
    }
}
