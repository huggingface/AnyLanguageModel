import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    @Suite("Ollama request history", .serialized)
    struct OllamaRequestHistoryTests {
        private func response(_ content: String) -> String {
            """
            {"model": "test", "created_at": "2026-10-03T00:00:00.000Z",
             "message": {"role": "assistant", "content": "\(content)"}, "done": true}
            """
        }

        @Test func nonstreamingRequestsSendInstructionsAndHistory() async throws {
            ReasoningURLProtocol.reset()
            ReasoningURLProtocol.enqueue(json: response("Hi"))
            ReasoningURLProtocol.enqueue(json: response("Hi again"))

            let model = OllamaLanguageModel(model: "test", session: ReasoningURLProtocol.makeSession())
            let session = LanguageModelSession(model: model, instructions: "Be brief.")
            _ = try await session.respond(to: "Hello")
            _ = try await session.respond(to: "Again")

            let body = try #require(ReasoningURLProtocol.recordedBodies.last)
            let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            let messages = try #require(json["messages"] as? [[String: Any]])
            #expect(messages.map { $0["role"] as? String } == ["system", "user", "assistant", "user"])
            #expect(messages.map { $0["content"] as? String } == ["Be brief.", "Hello", "Hi", "Again"])
        }
    }
#endif
