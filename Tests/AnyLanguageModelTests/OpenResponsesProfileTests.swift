import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        @Test(arguments: [false, true])
        func openResponsesReusedCallIDs(_ streaming: Bool) async throws {
            for profile in [false, true] {
                try await providerReusedCallIDsPreserveEveryOutput(
                    .openResponses,
                    streaming: streaming,
                    profile: profile
                )
            }
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesProfileReevaluation(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileReevaluatesBeforeEveryToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesModelHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerProfileCanSwitchModelForToolContinuation(provider, streaming)
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesSchemaPromptHandoff(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsRestoreOriginalSchemaPromptOption(provider, streaming)
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesHistoryAcrossHandoffs(_ provider: Provider, _ streaming: Bool) async throws {
            try await modelHandoffsPreserveEarlierToolResultsAndHistoryWrites(provider, streaming)
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesSharedToolRoundBudget(_ provider: Provider, _ streaming: Bool) async throws {
            try await providerModelHandoffsShareToolRoundLimit(provider, streaming)
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesFailedHandoffHistory(_ provider: Provider, _ streaming: Bool) async throws {
            try await failedHandoffDoesNotPersistPendingHistory(provider, streaming)
        }

        @Test(arguments: [Provider.openResponses], [false, true])
        func openResponsesOrdinaryToolLoop(_ provider: Provider, _ streaming: Bool) async throws {
            try await ordinaryWrapperKeepsTheProviderToolLoop(provider, streaming)
        }
    }
#endif
