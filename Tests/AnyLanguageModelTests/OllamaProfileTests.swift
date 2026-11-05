import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        @Test(arguments: [false, true], [false, true])
        func ollamaNonStreamingReportsFinalRequestUsage(profile: Bool, finalReportsUsage: Bool) async throws {
            let provider = Provider.ollama
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try Self.json(provider.response(text: "", counts: provider.counts, tool: true))
            )
            UsageURLProtocol.enqueue(
                json: try Self.json(
                    provider.response(
                        text: "Done",
                        counts: finalReportsUsage ? ["prompt_eval_count": 3, "eval_count": 2] : nil
                    )
                )
            )
            let session =
                profile
                ? LanguageModelSession(
                    profile: LanguageModelSession.Profile { WeatherTool() }.model(provider.makeModel())
                )
                : provider.makeSession(tools: [WeatherTool()])
            let response = try await session.respond(to: "Weather?")
            let finalUsage: LanguageModelSession.Usage =
                finalReportsUsage
                ? .init(
                    input: .init(totalTokenCount: 3, cachedTokenCount: 0),
                    output: .init(totalTokenCount: 2, reasoningTokenCount: 0)
                ) : .zero
            var everyRequest = provider.expected
            everyRequest.add(finalUsage)
            #expect(response.content == "Done")
            #expect(response.usage == finalUsage)
            #expect(session.usage == everyRequest)
            #expect(UsageURLProtocol.recordedBodies.count == 2)
        }

        @Test(arguments: [false, true])
        func ollamaReusedCallIDs(_ streaming: Bool) async throws {
            for profile in [false, true] {
                try await providerReusedCallIDsPreserveEveryOutput(.ollama, streaming: streaming, profile: profile)
            }
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaProfileReevaluation(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileReevaluatesBeforeEveryToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaModelHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileCanSwitchModelForToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaSchemaPromptHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsRestoreOriginalSchemaPromptOption(provider, streaming)
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaHistoryAcrossHandoffs(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsPreserveEarlierToolResultsAndHistoryWrites(provider, streaming)
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaSharedToolRoundBudget(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerModelHandoffsShareToolRoundLimit(provider, streaming)
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaFailedHandoffHistory(_ provider: Provider, _ streaming: Bool) async throws {
            try await failedHandoffDoesNotPersistPendingHistory(provider, streaming)
        }

        @Test(arguments: [Provider.ollama], [false, true])
        func ollamaOrdinaryToolLoop(_ provider: Provider, _ streaming: Bool) async throws {
            try await ordinaryWrapperKeepsTheProviderToolLoop(provider, streaming)
        }
    }
#endif
