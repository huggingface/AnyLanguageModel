import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    import CoreGraphics

    @Suite("GeminiLanguageModel thought signatures", .serialized)
    struct GeminiThoughtSignatureTests {
        private static let signature = "CvsBAdHtim8n5xQK1pVX2H0lPQeXAMPLEsignature=="

        @Generable
        struct ImageArguments {
            var request: String
        }

        private struct ImageOutputTool: Tool {
            let name = "image"
            let description = "Return an image"
            let output: Prompt

            func call(arguments: ImageArguments) async throws -> Prompt {
                output
            }
        }

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

        private func image() throws -> CGImage {
            let context = try #require(
                CGContext(
                    data: nil,
                    width: 2,
                    height: 2,
                    bitsPerComponent: 8,
                    bytesPerRow: 8,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                )
            )
            return try #require(context.makeImage())
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

        @Test("preserves all text and image data in function responses")
        func preservesSegmentedToolOutput() async throws {
            StubURLProtocol.reset()
            StubURLProtocol.enqueue(
                json:
                    #"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"image","args":{"request":"read"}}}]},"finishReason":"STOP"}]}"#
            )
            StubURLProtocol.enqueue(json: textResponse("OK"))

            let output = try Prompt {
                "before"
                Attachment(try image())
                "after"
            }
            let session = LanguageModelSession(
                model: makeModel(),
                tools: [ImageOutputTool(output: output)]
            )
            _ = try await session.respond(to: "read image")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 2)
            let json = try #require(
                try JSONSerialization.jsonObject(with: bodies[1]) as? [String: Any]
            )
            let contents = try #require(json["contents"] as? [[String: Any]])
            let outputContent = try #require(contents.last)
            let outputParts = try #require(outputContent["parts"] as? [[String: Any]])
            let functionResponse = try #require(
                outputParts.first?["functionResponse"] as? [String: Any]
            )
            let response = try #require(functionResponse["response"] as? [String: Any])
            #expect(response["result"] as? String == "before\nafter")

            let mediaParts = try #require(functionResponse["parts"] as? [[String: Any]])
            let inlineData = try #require(mediaParts.first?["inlineData"] as? [String: Any])
            #expect(inlineData["mime_type"] as? String == "image/png")
            #expect(Data(base64Encoded: try #require(inlineData["data"] as? String)) != nil)
        }

        @Test("inlines URL-backed images in function responses")
        func inlinesURLBackedToolOutput() async throws {
            StubURLProtocol.reset()
            StubURLProtocol.enqueue(
                json:
                    #"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"image","args":{"request":"read"}}}]},"finishReason":"STOP"}]}"#
            )
            StubURLProtocol.enqueue(json: textResponse("OK"))

            let imageData = Data([0x89, 0x50, 0x4E, 0x47])
            let url = FileManager.default.temporaryDirectory
                .appending(path: UUID().uuidString)
                .appendingPathExtension("png")
            try imageData.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }

            let output = Prompt { Attachment(imageURL: url) }
            let session = LanguageModelSession(
                model: makeModel(),
                tools: [ImageOutputTool(output: output)]
            )
            _ = try await session.respond(to: "read image")

            let body = try #require(StubURLProtocol.recordedBodies.last)
            let json = try #require(
                try JSONSerialization.jsonObject(with: body) as? [String: Any]
            )
            let contents = try #require(json["contents"] as? [[String: Any]])
            let outputParts = try #require(contents.last?["parts"] as? [[String: Any]])
            let functionResponse = try #require(
                outputParts.first?["functionResponse"] as? [String: Any]
            )
            let mediaParts = try #require(functionResponse["parts"] as? [[String: Any]])
            let inlineData = try #require(mediaParts.first?["inlineData"] as? [String: Any])
            #expect(inlineData["mime_type"] as? String == "image/png")
            #expect(Data(base64Encoded: try #require(inlineData["data"] as? String)) == imageData)
            #expect(mediaParts.first?["fileData"] == nil)
        }

        @Test("sends remote Tool image URLs as ordered text without downloading", arguments: [false, true])
        func describesRemoteURLBackedToolOutput(streaming: Bool) async throws {
            StubURLProtocol.reset()
            let toolResponse =
                #"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"image","args":{"request":"read"}}}]},"finishReason":"STOP"}]}"#
            if streaming {
                StubURLProtocol.enqueue(eventStream: [toolResponse])
                StubURLProtocol.enqueue(eventStream: [try self.response(parts: [["text": "OK"]])])
            } else {
                StubURLProtocol.enqueue(json: toolResponse)
                StubURLProtocol.enqueue(json: textResponse("OK"))
            }

            let url = try #require(URL(string: "https://example.invalid/image.png"))
            let output = Prompt {
                "before"
                Attachment(imageURL: url)
                "after"
            }
            let session = LanguageModelSession(
                model: makeModel(),
                tools: [ImageOutputTool(output: output)]
            )

            let response =
                if streaming {
                    try await session.streamResponse(to: "read image").collect()
                } else {
                    try await session.respond(to: "read image")
                }
            #expect(response.content == "OK")
            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 2)

            func expectURLText(in body: Data) throws {
                let responses = try contents(in: body).flatMap { content in
                    (content["parts"] as? [[String: Any]] ?? []).compactMap {
                        $0["functionResponse"] as? [String: Any]
                    }
                }
                try #require(responses.count == 1)
                let functionResponse = try #require(responses.first)
                let result = try #require(functionResponse["response"] as? [String: Any])
                #expect(result["result"] as? String == "before\n\(url.absoluteString)\nafter")
                #expect(functionResponse["parts"] == nil)
            }
            try expectURLText(in: bodies[1])

            let outputs = session.transcript.compactMap { entry -> Transcript.ToolOutput? in
                if case .toolOutput(let output) = entry { return output }
                return nil
            }
            let segments = try #require(outputs.first).segments
            try #require(segments.count == 3)
            guard case .image(let image) = segments[1] else {
                Issue.record("The transcript must retain the remote image segment")
                return
            }
            #expect(image.source == .url(url))

            let data = try JSONEncoder().encode(session.transcript)
            let transcript = try JSONDecoder().decode(Transcript.self, from: data)
            #expect(transcript == session.transcript)
            let restored = LanguageModelSession(model: makeModel(), transcript: transcript)
            if streaming {
                StubURLProtocol.enqueue(eventStream: [try self.response(parts: [["text": "Replayed"]])])
                #expect(try await restored.streamResponse(to: "continue").collect().content == "Replayed")
            } else {
                StubURLProtocol.enqueue(json: textResponse("Replayed"))
                #expect(try await restored.respond(to: "continue").content == "Replayed")
            }
            let replayBodies = StubURLProtocol.recordedBodies
            try #require(replayBodies.count == 3)
            try expectURLText(in: replayBodies[2])
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

        private func response(parts: [[String: Any]]) throws -> String {
            let data = try JSONSerialization.data(withJSONObject: [
                "candidates": [
                    [
                        "content": ["role": "model", "parts": parts],
                        "finishReason": "STOP",
                    ]
                ]
            ])
            return String(decoding: data, as: UTF8.self)
        }

        private func contents(in body: Data) throws -> [[String: Any]] {
            let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
            return try #require(json["contents"] as? [[String: Any]])
        }

        private func expectParts(_ expected: [[String: Any]], in content: [String: Any]) throws {
            let actual = try #require(content["parts"] as? [[String: Any]])
            #expect(NSArray(array: actual).isEqual(to: expected))
        }

        private func restoredSession(_ session: LanguageModelSession) throws -> LanguageModelSession {
            let data = try JSONEncoder().encode(session.transcript)
            let transcript = try JSONDecoder().decode(Transcript.self, from: data)
            return LanguageModelSession(model: makeModel(), tools: [WeatherTool()], transcript: transcript)
        }

        @Test("round-trips semantic arguments through sequential calls and later turns")
        func roundTripsArguments() async throws {
            StubURLProtocol.reset()
            let arguments: [[String: Any]] = [
                [
                    "city": "Paris", "kind": "ordinary parameter", "id": "user-id",
                    "enabled": true, "count": 42, "ratio": 1.25, "nothing": NSNull(),
                    "nested": ["kind": "nested parameter", "items": ["hello", false, 2.5, NSNull(), ["x": 1]]],
                    "emptyObject": [String: String](), "emptyArray": [String](),
                ],
                ["city": "London", "kind": ["type": "user data"], "enabled": false, "count": -3],
            ]
            let signatures = [Self.signature, "second-signature=="]
            let callParts: [[String: Any]] = zip(arguments, signatures).map { args, signature in
                ["functionCall": ["name": "getWeather", "args": args], "thoughtSignature": signature]
            }
            for part in callParts {
                StubURLProtocol.enqueue(json: try response(parts: [part]))
            }
            StubURLProtocol.enqueue(json: textResponse("Both cities are sunny."))
            StubURLProtocol.enqueue(json: textResponse("Paris and London."))
            StubURLProtocol.enqueue(json: textResponse("Two cities."))

            let session = LanguageModelSession(model: makeModel(), tools: [WeatherTool()])
            _ = try await session.respond(to: "Check both cities")
            _ = try await session.respond(to: "Which cities?")
            let restored = try restoredSession(session)
            _ = try await restored.respond(to: "How many?")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 5)
            #expect(try bodies.map { try contentCount(in: $0) } == [1, 3, 5, 7, 9])
            for requestIndex in 1 ..< bodies.count {
                let history = try contents(in: bodies[requestIndex])
                try expectParts([callParts[0]], in: history[1])
                if requestIndex >= 2 {
                    try expectParts([callParts[1]], in: history[3])
                }
                #expect(
                    try functionCallSignatures(in: bodies[requestIndex])
                        == Array(signatures.prefix(min(requestIndex, 2)))
                )
            }
            let outputs = session.transcript.compactMap { entry -> String? in
                guard case .toolOutput(let output) = entry else { return nil }
                return output.segments.compactMap {
                    if case .text(let text) = $0 { return text.content }
                    if case .structure(let structure) = $0 { return structure.content.jsonString }
                    return nil
                }.joined()
            }
            #expect(outputs.count == 2)
            #expect(outputs.first?.contains("Paris") == true)
            #expect(outputs.last?.contains("London") == true)
        }

        @Test("preserves text signatures and part boundaries through saved conversation history")
        func roundTripsTextParts() async throws {
            StubURLProtocol.reset()
            let parts: [[String: Any]] = [
                ["text": "Thinking.", "thought": true, "thoughtSignature": "thinking-signature=="],
                ["text": "Hello "],
                ["text": "world.", "thoughtSignature": Self.signature],
            ]
            StubURLProtocol.enqueue(json: try response(parts: parts))
            StubURLProtocol.enqueue(json: textResponse("Again."))
            StubURLProtocol.enqueue(json: textResponse("Still here."))
            let session = LanguageModelSession(model: makeModel())
            let result = try await session.respond(to: "Hello")
            #expect(result.content == "Thinking.Hello world.")
            #expect(result.providerMetadata != nil)
            _ = try await session.respond(to: "Continue")
            let restored = try restoredSession(session)
            _ = try await restored.respond(to: "Continue again")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 3)
            #expect(try bodies.map { try contentCount(in: $0) } == [1, 3, 5])
            for body in bodies.dropFirst() {
                try expectParts(parts, in: contents(in: body)[1])
            }
        }

        @Test("preserves signed text siblings in a tool-call turn")
        func roundTripsMixedParts() async throws {
            StubURLProtocol.reset()
            let parts: [[String: Any]] = [
                ["text": "Checking.", "thoughtSignature": "before-call=="],
                [
                    "functionCall": ["name": "getWeather", "args": ["city": "Paris"]],
                    "thoughtSignature": Self.signature,
                ],
                ["text": "Then London."],
                ["functionCall": ["name": "getWeather", "args": ["city": "London"]]],
                ["text": "Waiting.", "thoughtSignature": "after-call=="],
            ]
            StubURLProtocol.enqueue(json: try response(parts: parts))
            StubURLProtocol.enqueue(json: textResponse("Sunny."))
            StubURLProtocol.enqueue(json: textResponse("Both cities."))
            let session = LanguageModelSession(model: makeModel(), tools: [WeatherTool()])
            _ = try await session.respond(to: "Check the weather")
            let restored = try restoredSession(session)
            _ = try await restored.respond(to: "Where?")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 3)
            #expect(try bodies.map { try contentCount(in: $0) } == [1, 4, 6])
            for body in bodies.dropFirst() {
                try expectParts(parts, in: contents(in: body)[1])
            }
        }

        @Test("preserves signed structured responses")
        func roundTripsStructuredResponse() async throws {
            StubURLProtocol.reset()
            let parts: [[String: Any]] = [
                ["text": #"{"city": "Paris"}"#, "thoughtSignature": Self.signature]
            ]
            StubURLProtocol.enqueue(json: try response(parts: parts))
            StubURLProtocol.enqueue(json: textResponse("Paris."))
            let session = LanguageModelSession(model: makeModel())
            let result = try await session.respond(to: "Pick a city", generating: WeatherTool.Arguments.self)
            #expect(result.content.city == "Paris")
            _ = try await session.respond(to: "Which city?")
            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 2)
            try expectParts(parts, in: contents(in: bodies[1])[1])
        }

        @Test("preserves a streaming signature arriving after the final text", arguments: [true, false])
        func roundTripsStreamingSignature(omitsText: Bool) async throws {
            StubURLProtocol.reset()
            var signaturePart: [String: Any] = ["thoughtSignature": Self.signature]
            if !omitsText { signaturePart["text"] = "" }
            StubURLProtocol.enqueue(
                eventStream: try [
                    response(parts: [["text": "Hello "]]),
                    response(parts: [["text": "world."]]),
                    response(parts: [signaturePart]),
                ]
            )
            StubURLProtocol.enqueue(json: textResponse("Again."))
            let session = LanguageModelSession(model: makeModel())
            let result = try await session.streamResponse(to: "Hello").collect()
            #expect(result.content == "Hello world.")
            #expect(result.providerMetadata != nil)
            let restored = try restoredSession(session)
            _ = try await restored.respond(to: "Continue")

            let bodies = StubURLProtocol.recordedBodies
            try #require(bodies.count == 2)
            #expect(try contentCount(in: bodies[1]) == 3)
            try expectParts(
                [
                    ["text": "Hello "], ["text": "world."],
                    ["text": "", "thoughtSignature": Self.signature],
                ],
                in: contents(in: bodies[1])[1]
            )
        }

    }

#endif
