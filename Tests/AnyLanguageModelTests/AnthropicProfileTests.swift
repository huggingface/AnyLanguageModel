import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        @Test(arguments: [false, true])
        func anthropicProfileUnsupportedToolImageUsesPlaceholder(_ streaming: Bool) async throws {
            let provider = Provider.anthropic
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            UsageURLProtocol.enqueue(json: try streaming ? provider.stream() : Self.json(provider.response()))
            let session = LanguageModelSession(
                profile: LanguageModelSession.Profile { WeatherTool() }.model(provider.makeModel())
            )
            let segments: [Transcript.Segment] = [
                .text(.init(content: "before")),
                .image(.init(data: testImageData, mimeType: "image/tiff")),
                .text(.init(content: "after")),
            ]
            session.toolExecutionDelegate = ProfileImageDelegate(segments: segments)
            if streaming {
                _ = try await session.streamResponse(to: "Weather?").collect()
            } else {
                _ = try await session.respond(to: "Weather?")
            }
            try #require(UsageURLProtocol.recordedBodies.count == 2)
            let request = try JSONDecoder().decode(
                [String: JSONValue].self,
                from: try #require(UsageURLProtocol.recordedBodies.last)
            )
            let content =
                request["messages"]?.arrayValue?.flatMap { $0.objectValue?["content"]?.arrayValue ?? [] }
                .first { $0.objectValue?["type"] == .string("tool_result") }?.objectValue?["content"]?.arrayValue ?? []
            #expect(content.compactMap { $0.objectValue?["text"]?.stringValue } == ["before", "<image>", "after"])
            #expect(content.allSatisfy { $0.objectValue?["type"] == .string("text") })
            let output = session.transcript.compactMap { entry -> Transcript.ToolOutput? in
                if case .toolOutput(let output) = entry { return output }
                return nil
            }.first
            #expect(output?.segments == segments)
        }

        private struct ProfileImageDelegate: ToolExecutionDelegate {
            let segments: [Transcript.Segment]

            func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
                -> ToolExecutionDecision
            { .provideOutput(segments) }
        }

        @Test(arguments: [false, true])
        func anthropicReusedCallIDs(_ streaming: Bool) async throws {
            for profile in [false, true] {
                try await providerReusedCallIDsPreserveEveryOutput(.anthropic, streaming: streaming, profile: profile)
            }
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicProfileReevaluation(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileReevaluatesBeforeEveryToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicModelHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileCanSwitchModelForToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicSchemaPromptHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsRestoreOriginalSchemaPromptOption(provider, streaming)
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicHistoryAcrossHandoffs(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsPreserveEarlierToolResultsAndHistoryWrites(provider, streaming)
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicSharedToolRoundBudget(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerModelHandoffsShareToolRoundLimit(provider, streaming)
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicFailedHandoffHistory(_ provider: Provider, _ streaming: Bool) async throws {
            try await failedHandoffDoesNotPersistPendingHistory(provider, streaming)
        }

        @Test(arguments: [Provider.anthropic], [false, true])
        func anthropicOrdinaryToolLoop(_ provider: Provider, _ streaming: Bool) async throws {
            if streaming {
                try await ordinaryWrapperKeepsTheProviderToolLoop(provider, streaming)
            } else {
                UsageURLProtocol.reset()
                UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "", tool: true)))
                let session = provider.makeSession(tools: [WeatherTool()])
                let response = try await session.respond(to: "Weather?")
                #expect(response.content.isEmpty)
                #expect(UsageURLProtocol.recordedBodies.count == 1)
                #expect(response.transcriptEntries.count == 2)
                #expect(session.transcript.count == 4)
            }
        }
    }
#endif
