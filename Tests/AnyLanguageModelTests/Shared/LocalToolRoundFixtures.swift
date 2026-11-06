import Foundation
import Testing

@testable import AnyLanguageModel

#if MLX || Llama
    /// Gives the code that `UnlockTool` needs, so a model has to call the tools in separate rounds.
    private struct GetCodeTool: Tool {
        let name = "getCode"
        let description = "Gets the code that unlocks a door"

        @Generable
        struct Arguments {
            @Guide(description: "The name of the door")
            var door: String
        }

        func call(arguments: Arguments) async throws -> String {
            "The code is 4217. Call unlock with the code 4217."
        }
    }

    private struct UnlockTool: Tool {
        let name = "unlock"
        let description = "Unlocks a door with the code from getCode"

        @Generable
        struct Arguments {
            @Guide(description: "The code from getCode")
            var code: String
        }

        func call(arguments: Arguments) async throws -> String {
            "The door is open."
        }
    }

    private func toolCallsIDs(_ entries: some Sequence<Transcript.Entry>) -> [String] {
        entries.compactMap { entry in
            if case .toolCalls(let calls) = entry { return calls.id }
            return nil
        }
    }

    private let doorInstructions = """
        You open doors. First call getCode. Then call unlock with the code that getCode returns. \
        Never guess the code. After unlock returns, reply with one short sentence.
        """

    /// Checks that a tool round has one entry across a response whose callbacks write `history`,
    /// in what the callbacks read, in the response, and in the session's transcript.
    func expectOneEntryForEachToolRound(
        model: some LanguageModel,
        streaming: Bool,
        prompt: String,
        options: GenerationOptions
    ) async throws {
        let seenByCallbacks = Locked<[[String]]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile {
                Instructions(doorInstructions)
                GetCodeTool()
                UnlockTool()
            }
            .model(model)
            .onToolOutput { _, _ in
                guard let properties = SessionPropertyBinding.values else {
                    Issue.record("Expected session properties in the callback")
                    return
                }
                let history = properties.history
                seenByCallbacks.withLock { $0.append(toolCallsIDs(history)) }
                properties.history = history
            }
        )

        let response =
            try await streaming
            ? session.streamResponse(to: prompt, options: options).collect()
            : session.respond(to: prompt, options: options)
        attachResponse(response)

        let responseRounds = toolCallsIDs(response.transcriptEntries)
        try #require(responseRounds.count >= 2, "The model made \(responseRounds.count) tool rounds")
        #expect(seenByCallbacks.withLock { $0.last } == responseRounds)
        #expect(toolCallsIDs(session.transcript) == responseRounds)
    }

    /// Checks that the tool round a delegate stops is the entry that the callbacks read.
    func expectStoppedToolRoundKeepsItsEntry(
        model: some LanguageModel,
        streaming: Bool,
        prompt: String,
        options: GenerationOptions
    ) async throws {
        let seenByCallbacks = Locked<[[String]]>([])
        let session = LanguageModelSession(
            profile: LanguageModelSession.Profile {
                Instructions(doorInstructions)
                GetCodeTool()
                UnlockTool()
            }
            .model(model)
            .onToolCall { _ in
                guard let properties = SessionPropertyBinding.values else {
                    Issue.record("Expected session properties in the callback")
                    return
                }
                let history = properties.history
                seenByCallbacks.withLock { $0.append(toolCallsIDs(history)) }
            }
        )
        session.toolExecutionDelegate = StopToolExecutionDelegate()

        let response =
            try await streaming
            ? session.streamResponse(to: prompt, options: options).collect()
            : session.respond(to: prompt, options: options)
        attachResponse(response)

        let responseRounds = toolCallsIDs(response.transcriptEntries)
        try #require(responseRounds.count == 1, "The model made \(responseRounds.count) tool rounds")
        #expect(seenByCallbacks.withLock { $0 } == [responseRounds])
    }
#endif
