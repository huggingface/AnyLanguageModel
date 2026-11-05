import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        @Test(arguments: [Provider.gemini], [false, true])
        func geminiProfileReevaluation(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileReevaluatesBeforeEveryToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.gemini], [false, true])
        func geminiModelHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileCanSwitchModelForToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.gemini], [false, true])
        func geminiSchemaPromptHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsRestoreOriginalSchemaPromptOption(provider, streaming)
        }

        @Test(arguments: [Provider.gemini], [false, true])
        func geminiHistoryAcrossHandoffs(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsPreserveEarlierToolResultsAndHistoryWrites(provider, streaming)
        }

        @Test(arguments: [Provider.gemini], [false, true])
        func geminiSharedToolRoundBudget(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerModelHandoffsShareToolRoundLimit(provider, streaming)
        }

        @Test(arguments: [Provider.gemini], [false, true])
        func geminiFailedHandoffHistory(_ provider: Provider, _ streaming: Bool) async throws {
            try await failedHandoffDoesNotPersistPendingHistory(provider, streaming)
        }

        @Test(arguments: [Provider.gemini], [false, true])
        func geminiOrdinaryToolLoop(_ provider: Provider, _ streaming: Bool) async throws {
            try await ordinaryWrapperKeepsTheProviderToolLoop(provider, streaming)
        }

        @Test(arguments: [false, true])
        func geminiProfileSaysSchemaAfterPromptInEveryRequest(_ streaming: Bool) async throws {
            let provider = Provider.gemini
            let final = #"{"answer":"OK"}"#
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: final) : Self.json(provider.response(text: final))
            )
            let session = LanguageModelSession(
                profile: LanguageModelSession.Profile { WeatherTool() }.model(provider.makeModel())
            )
            let content =
                try await streaming
                ? session.streamResponse(to: "Weather?", generating: GeminiProfileAnswer.self).collect().content
                : session.respond(to: "Weather?", generating: GeminiProfileAnswer.self).content
            #expect(content.answer == "OK")

            let bodies = UsageURLProtocol.recordedBodies
            try #require(bodies.count == 2)
            for body in bodies {
                let contents = try geminiContents(in: body)
                let prompt = try #require(contents.first?.objectValue?["parts"]?.arrayValue)
                let texts = prompt.compactMap { $0.objectValue?["text"]?.stringValue }
                try #require(texts.count == 2)
                #expect(texts[0] == "Weather?")
                #expect(texts[1].hasPrefix("Respond with valid JSON matching this object schema"))
            }
            let continuation = try geminiContents(in: bodies[1])
            #expect(continuation.count == 3)
            #expect(try geminiFunctionResponses(in: bodies[1]).count == 1)
        }

        @Test(arguments: [false, true])
        func geminiProfileSendsStructuredToolOutputAsPlainJSON(_ streaming: Bool) async throws {
            let response = try await geminiProfileFunctionResponse(
                streaming: streaming,
                segments: [.structure(.init(source: "getWeather", content: GeneratedContent(properties: ["count": 3])))]
            )
            #expect(response["response"] == .object(["result": .object(["count": .int(3)])]))
        }

        @Test(arguments: [false, true])
        func geminiProfileUnsupportedToolImageUsesPlaceholder(_ streaming: Bool) async throws {
            let response = try await geminiProfileFunctionResponse(
                streaming: streaming,
                segments: [
                    .text(.init(content: "before")),
                    .image(.init(data: testImageData, mimeType: "image/tiff")),
                    .text(.init(content: "after")),
                ]
            )
            #expect(response["response"] == .object(["result": .string("before\n<image>\nafter")]))
            #expect(response["parts"] == nil)
        }

        @Test(arguments: [false, true])
        func geminiProfileRemoteToolImageUsesPlainText(_ streaming: Bool) async throws {
            let url = try #require(URL(string: "https://example.invalid/image.png"))
            let response = try await geminiProfileFunctionResponse(
                streaming: streaming,
                segments: [
                    .text(.init(content: "before")),
                    .image(.init(source: .url(url))),
                    .text(.init(content: "after")),
                ]
            )
            #expect(response["response"] == .object(["result": .string("before\n\(url.absoluteString)\nafter")]))
            #expect(response["parts"] == nil)
        }

        @Test(arguments: [false, true])
        func geminiProfileSupportedToolImageIsInlined(_ streaming: Bool) async throws {
            let response = try await geminiProfileFunctionResponse(
                streaming: streaming,
                segments: [.image(.init(data: testImageData, mimeType: "image/png"))]
            )
            let part = try #require(response["parts"]?.arrayValue?.first?.objectValue?["inlineData"]?.objectValue)
            #expect(part["mime_type"] == .string("image/png"))
            #expect(part["data"] == .string(testImageData.base64EncodedString()))
        }

        @Test(arguments: [false, true])
        func geminiProfileRetainsThoughtSignature(_ streaming: Bool) async throws {
            let provider = Provider.gemini
            let signature = "profile-signature"
            UsageURLProtocol.reset()
            let toolResponse =
                try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            UsageURLProtocol.enqueue(
                json: toolResponse.replacingOccurrences(
                    of: "\"functionCall\":",
                    with: "\"thoughtSignature\":\"\(signature)\",\"functionCall\":"
                )
            )
            UsageURLProtocol.enqueue(json: try streaming ? provider.stream() : Self.json(provider.response()))
            let session = LanguageModelSession(
                profile: LanguageModelSession.Profile { WeatherTool() }.model(provider.makeModel())
            )
            if streaming {
                _ = try await session.streamResponse(to: "Weather?").collect()
            } else {
                _ = try await session.respond(to: "Weather?")
            }
            let body = try #require(UsageURLProtocol.recordedBodies.last)
            let call = try geminiContents(in: body)
                .flatMap { $0.objectValue?["parts"]?.arrayValue ?? [] }
                .first { $0.objectValue?["functionCall"] != nil }
            #expect(call?.objectValue?["thoughtSignature"] == .string(signature))
        }

        private func geminiProfileFunctionResponse(
            streaming: Bool,
            segments: [Transcript.Segment]
        ) async throws -> [String: JSONValue] {
            let provider = Provider.gemini
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            UsageURLProtocol.enqueue(json: try streaming ? provider.stream() : Self.json(provider.response()))
            let session = LanguageModelSession(
                profile: LanguageModelSession.Profile { WeatherTool() }.model(provider.makeModel())
            )
            session.toolExecutionDelegate = GeminiProfileOutputDelegate(segments: segments)
            if streaming {
                _ = try await session.streamResponse(to: "Weather?").collect()
            } else {
                _ = try await session.respond(to: "Weather?")
            }
            try #require(UsageURLProtocol.recordedBodies.count == 2)
            let output = session.transcript.compactMap { entry -> Transcript.ToolOutput? in
                if case .toolOutput(let output) = entry { return output }
                return nil
            }.first
            #expect(output?.segments == segments)
            let responses = try geminiFunctionResponses(in: try #require(UsageURLProtocol.recordedBodies.last))
            try #require(responses.count == 1)
            return try #require(responses.first)
        }

        private func geminiContents(in body: Data) throws -> [JSONValue] {
            let request = try JSONDecoder().decode([String: JSONValue].self, from: body)
            return try #require(request["contents"]?.arrayValue)
        }

        private func geminiFunctionResponses(in body: Data) throws -> [[String: JSONValue]] {
            try geminiContents(in: body)
                .flatMap { $0.objectValue?["parts"]?.arrayValue ?? [] }
                .compactMap { $0.objectValue?["functionResponse"]?.objectValue }
        }
    }

    @Generable
    private struct GeminiProfileAnswer {
        var answer: String
    }

    private struct GeminiProfileOutputDelegate: ToolExecutionDelegate {
        let segments: [Transcript.Segment]

        func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
            -> ToolExecutionDecision
        { .provideOutput(segments) }
    }
#endif
