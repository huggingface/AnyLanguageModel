import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)

    @Suite("GeminiLanguageModel thought signatures", .serialized)
    struct GeminiThoughtSignatureTests {
        private static let signature = "CvsBAdHtim8n5xQK1pVX2H0lPQeXAMPLEsignature=="

        private func makeModel() -> GeminiLanguageModel {
            GeminiLanguageModel(
                apiKey: "test-key",
                model: "gemini-3.6-flash",
                session: StubURLProtocol.makeSession()
            )
        }

        private func functionCallResponse(signature: String?) -> String {
            let signatureField = signature.map { ", \"thoughtSignature\": \"\($0)\"" } ?? ""
            return """
                {
                  "candidates": [
                    {
                      "content": {
                        "role": "model",
                        "parts": [
                          {
                            "functionCall": { "name": "getWeather", "args": { "city": "Paris" } }\(signatureField)
                          }
                        ]
                      },
                      "finishReason": "STOP"
                    }
                  ]
                }
                """
        }

        private func textResponse(_ text: String) -> String {
            """
            {
              "candidates": [
                {
                  "content": { "role": "model", "parts": [{ "text": "\(text)" }] },
                  "finishReason": "STOP"
                }
              ]
            }
            """
        }

        /// Signatures of every `functionCall` part in a request body, in order.
        /// A part without a signature contributes `nil`.
        private func functionCallSignatures(in body: Data) throws -> [String?] {
            let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
            let contents = json?["contents"] as? [[String: Any]] ?? []
            return contents.flatMap { content -> [String?] in
                let parts = content["parts"] as? [[String: Any]] ?? []
                return parts.compactMap { part -> String?? in
                    guard part["functionCall"] != nil else { return nil }
                    return .some(part["thoughtSignature"] as? String)
                }
            }
        }

        /// The number of `contents` entries in a request body.
        private func contentCount(in body: Data) throws -> Int {
            let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
            return (json?["contents"] as? [[String: Any]] ?? []).count
        }

        @Test("echoes the thought signature back with the function results")
        func echoesSignatureOnFollowUpRequest() async throws {
            StubURLProtocol.reset()
            StubURLProtocol.enqueue(json: functionCallResponse(signature: Self.signature))
            StubURLProtocol.enqueue(json: textResponse("It is sunny in Paris."))

            let session = LanguageModelSession(model: makeModel(), tools: [WeatherTool()])
            let response = try await session.respond(to: "What is the weather in Paris?")

            #expect(response.content == "It is sunny in Paris.")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 2)
            #expect(try functionCallSignatures(in: bodies[1]) == [Self.signature])
        }

        @Test("keeps the signature on the tool call when the conversation continues")
        func keepsSignatureOnLaterTurn() async throws {
            StubURLProtocol.reset()
            StubURLProtocol.enqueue(json: functionCallResponse(signature: Self.signature))
            StubURLProtocol.enqueue(json: textResponse("It is sunny in Paris."))
            StubURLProtocol.enqueue(json: textResponse("You asked about the weather in Paris."))

            let session = LanguageModelSession(model: makeModel(), tools: [WeatherTool()])
            _ = try await session.respond(to: "What is the weather in Paris?")
            _ = try await session.respond(to: "What did I just ask about?")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 3)

            // The tool call is replayed as history on the next turn, and Gemini rejects a request
            // whose functionCall parts have lost their signatures.
            let signatures = try functionCallSignatures(in: bodies[2])
            #expect(!signatures.isEmpty)
            #expect(signatures.allSatisfy { $0 == Self.signature })
        }

        @Test("does not replay the conversation history on later turns")
        func doesNotReplayHistoryOnLaterTurns() async throws {
            StubURLProtocol.reset()
            StubURLProtocol.enqueue(json: functionCallResponse(signature: Self.signature))
            StubURLProtocol.enqueue(json: textResponse("It is sunny in Paris."))
            StubURLProtocol.enqueue(json: textResponse("You asked about the weather."))
            StubURLProtocol.enqueue(json: textResponse("Paris, specifically."))

            let session = LanguageModelSession(model: makeModel(), tools: [WeatherTool()])
            _ = try await session.respond(to: "What is the weather in Paris?")
            _ = try await session.respond(to: "What did I just ask about?")
            _ = try await session.respond(to: "Which city was that?")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 4)

            // One entry per turn, plus the tool call and its output. A response that reported the
            // whole transcript instead of its own entries would compound: 1, 3, 6, 14.
            #expect(try bodies.map { try contentCount(in: $0) } == [1, 3, 5, 7])

            // The tool call is replayed once, still signed, however many turns later.
            #expect(try functionCallSignatures(in: bodies[3]) == [Self.signature])
        }
    }

#endif
