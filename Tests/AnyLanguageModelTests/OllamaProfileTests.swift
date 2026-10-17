import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
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
