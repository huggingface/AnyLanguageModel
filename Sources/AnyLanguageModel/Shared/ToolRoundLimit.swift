import Foundation

/// Bounds the tool rounds in one response from a remote provider.
///
/// A model that keeps calling tools would otherwise run tools
/// and send requests until the caller cancels.
/// Like the MLX and llama.cpp providers,
/// a response allows at most ``maximumRounds`` rounds of tool calls
/// and stops when a round repeats the previous round's calls exactly.
final class ToolRoundLimit: @unchecked Sendable {
    /// A tool call, reduced to what identifies a repeated round.
    struct Call: Equatable {
        var name: String
        var arguments: JSONValue?

        init(name: String, arguments: JSONValue?) {
            self.name = name
            self.arguments = arguments
        }

        init(name: String, arguments: [String: JSONValue]?) {
            self.init(name: name, arguments: arguments.map(JSONValue.object))
        }

        /// Creates a call from arguments encoded as a JSON string.
        ///
        /// Arguments that aren't valid JSON are compared as a string.
        init(name: String, jsonArguments: String?) {
            let arguments = jsonArguments.map { json in
                (try? JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))) ?? .string(json)
            }
            self.init(name: name, arguments: arguments)
        }
    }

    /// The maximum number of tool rounds in one response.
    static let maximumRounds = 8

    /// The provider name used in error messages.
    let provider: String

    private let state = Locked(State())

    init(provider: String) {
        self.provider = provider
    }

    /// Records a round of tool calls before any of them runs.
    ///
    /// - Throws: A decoding failure
    ///   when the round exceeds ``maximumRounds``
    ///   or repeats the previous round's calls.
    func record(_ calls: [Call]) throws {
        try state.withLock { state in
            state.rounds += 1
            if state.rounds > Self.maximumRounds {
                throw LanguageModelSession.GenerationError.decodingFailure(
                    .init(
                        debugDescription:
                            "Exceeded maximum tool iterations (\(Self.maximumRounds)) while processing \(provider) tool calls."
                    )
                )
            }
            if calls == state.previousCalls {
                throw LanguageModelSession.GenerationError.decodingFailure(
                    .init(
                        debugDescription:
                            "Detected repeated \(provider) tool-call signature and aborted to avoid an infinite tool loop."
                    )
                )
            }
            state.previousCalls = calls
        }
    }

    private struct State {
        var rounds = 0
        var previousCalls: [Call]?
    }
}
