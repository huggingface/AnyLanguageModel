import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        @Test func anthropicRespondContinuesWithToolOutputForText() async throws {
            let session = try anthropicToolThenAnswerSession(answer: "Sunny")
            let response = try await session.respond(to: "Weather?")
            #expect(response.content == "Sunny")
            #expect(response.transcriptEntries.count == 2)
            try expectAnthropicContinuation(of: session)
        }

        @Test func anthropicRespondContinuesWithToolOutputForStructuredContent() async throws {
            let session = try anthropicToolThenAnswerSession(answer: #"{"answer":"Sunny"}"#)
            let response = try await session.respond(to: "Weather?", generating: Answer.self)
            #expect(response.content.answer == "Sunny")
            #expect(response.transcriptEntries.count == 2)
            try expectAnthropicContinuation(of: session)
        }

        private func anthropicToolThenAnswerSession(answer: String) throws -> LanguageModelSession {
            let provider = Provider.anthropic
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try Self.json(provider.response(text: "", counts: provider.counts, tool: true))
            )
            UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: answer, counts: provider.counts)))
            return provider.makeSession(tools: [WeatherTool()])
        }

        private func expectAnthropicContinuation(of session: LanguageModelSession) throws {
            let bodies = UsageURLProtocol.recordedBodies
            try #require(bodies.count == 2)

            let body = try #require(try JSONSerialization.jsonObject(with: bodies[1]) as? [String: Any])
            let messages = try #require(body["messages"] as? [[String: Any]])
            let call = try #require((messages[messages.count - 2]["content"] as? [[String: Any]])?.first)
            #expect(messages[messages.count - 2]["role"] as? String == "assistant")
            #expect(call["type"] as? String == "tool_use")
            #expect(call["id"] as? String == "call_1")

            let result = try #require((messages[messages.count - 1]["content"] as? [[String: Any]])?.first)
            let output = try #require((result["content"] as? [[String: Any]])?.first)
            #expect(messages[messages.count - 1]["role"] as? String == "user")
            #expect(result["type"] as? String == "tool_result")
            #expect(result["tool_use_id"] as? String == "call_1")
            #expect(output["text"] as? String == "The weather in Paris is sunny and 72°F / 23°C")

            var everyRequest = Provider.anthropic.expected
            everyRequest.add(Provider.anthropic.expected)
            #expect(session.usage == everyRequest)
        }
    }
#endif
