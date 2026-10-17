import Foundation

/// Bounds the tool rounds in one response.
///
/// A model that keeps calling tools would otherwise run tools
/// and send requests until the caller cancels.
/// A response allows at most ``maximumRounds`` rounds of tool calls.
/// A round that repeats the previous one runs like any other:
/// calling the same tool with the same arguments again can be what's asked,
/// and the limit is what ends a loop.
final class ToolRoundLimit: @unchecked Sendable {
    /// The maximum number of tool rounds in one response.
    static let maximumRounds = 8

    /// The provider name used in error messages.
    let provider: String

    private let roundCount = Locked(0)

    init(provider: String) {
        self.provider = provider
    }

    /// Records a round of tool calls before any of them runs.
    ///
    /// - Throws: A decoding failure when the round exceeds ``maximumRounds``.
    func record() throws {
        try roundCount.withLock { rounds in
            rounds += 1
            if rounds > Self.maximumRounds {
                throw LanguageModelSession.GenerationError.decodingFailure(
                    .init(
                        debugDescription:
                            "Exceeded maximum tool iterations (\(Self.maximumRounds)) while processing \(provider) tool calls."
                    )
                )
            }
        }
    }
}
