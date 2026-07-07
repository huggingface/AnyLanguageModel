import Foundation
import Testing

@testable import AnyLanguageModel

#if LiteRT
    /// Path to a local `.litertlm` file to test against
    /// (for example, gemma-4-E2B-it.litertlm).
    /// These tests load multi-GB weights, so they only run when explicitly
    /// requested via the `LITERT_TEST_MODEL` environment variable.
    private let liteRTTestModelPath = ProcessInfo.processInfo.environment["LITERT_TEST_MODEL"]

    private let shouldRunLiteRTTests = liteRTTestModelPath != nil

    @Generable
    private struct CityAnswer {
        var city: String
    }

    @Suite("LiteRTLanguageModel", .enabled(if: shouldRunLiteRTTests), .serialized)
    struct LiteRTLanguageModelTests {
        private var model: LiteRTLanguageModel {
            LiteRTLanguageModel(
                modelFileURL: URL(fileURLWithPath: liteRTTestModelPath!),
                modalities: []
            )
        }

        @Test func respondToTextPrompt() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(to: "Reply with a single word: hello")
            #expect(!response.content.isEmpty)
        }

        @Test func streamTextPrompt() async throws {
            let session = LanguageModelSession(model: model)
            var snapshots = 0
            var lastContent = ""
            for try await snapshot in session.streamResponse(to: "Count from 1 to 5.") {
                snapshots += 1
                lastContent = snapshot.content
            }
            #expect(snapshots > 1)
            #expect(!lastContent.isEmpty)
        }

        @Test func structuredGeneration() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "What is the capital of France?",
                generating: CityAnswer.self
            )
            #expect(!response.content.city.isEmpty)
        }

        @Test func multiTurnConversationKeepsContext() async throws {
            let session = LanguageModelSession(model: model)
            _ = try await session.respond(to: "My name is Alice. Remember it.")
            let response = try await session.respond(to: "What is my name?")
            #expect(response.content.localizedCaseInsensitiveContains("alice"))
        }
    }
#endif
