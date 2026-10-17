import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAIProfileReevaluation(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileReevaluatesBeforeEveryToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAIModelHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileCanSwitchModelForToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAISchemaPromptHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsRestoreOriginalSchemaPromptOption(provider, streaming)
        }

        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAIHistoryAcrossHandoffs(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsPreserveEarlierToolResultsAndHistoryWrites(provider, streaming)
        }

        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAISharedToolRoundBudget(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerModelHandoffsShareToolRoundLimit(provider, streaming)
        }

        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAIFailedHandoffHistory(_ provider: Provider, _ streaming: Bool) async throws {
            try await failedHandoffDoesNotPersistPendingHistory(provider, streaming)
        }

        @Test(arguments: [Provider.chat, Provider.responses], [false, true])
        func openAIOrdinaryToolLoop(_ provider: Provider, _ streaming: Bool) async throws {
            try await ordinaryWrapperKeepsTheProviderToolLoop(provider, streaming)
        }
    }
#endif
