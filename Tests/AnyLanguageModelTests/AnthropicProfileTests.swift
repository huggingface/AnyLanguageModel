import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
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
