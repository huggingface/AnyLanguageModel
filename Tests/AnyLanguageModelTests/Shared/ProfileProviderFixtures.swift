import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        func providerProfileReevaluatesBeforeEveryToolContinuation(
            _ provider: Provider,
            _ streaming: Bool
        ) async throws {
            UsageURLProtocol.reset()
            if streaming {
                UsageURLProtocol.enqueue(json: try provider.toolStream())
                UsageURLProtocol.enqueue(json: try provider.stream(text: "Profile done"))
            } else {
                UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "", tool: true)))
                UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "Profile done")))
            }
            let model = provider.makeModel()
            let state = ProviderProfileState()
            let session = LanguageModelSession(
                profile: ProviderProfile(state: state, model: model)
            )

            let response =
                try await streaming
                ? session.streamResponse(to: "Weather?").collect()
                : session.respond(to: "Weather?")

            #expect(response.content == "Profile done")
            #expect(state.executions.withLock { $0 } == ["Paris"])
            #expect(state.transformSawOutput.withLock { $0 } == [false, true])
            #expect(
                state.events.withLock { $0 } == [
                    "call:A", "execute:A", "output:A",
                ]
            )
            #expect(UsageURLProtocol.recordedBodies.count == 2)
            let first = String(decoding: UsageURLProtocol.recordedBodies[0], as: UTF8.self)
            let second = String(decoding: UsageURLProtocol.recordedBodies[1], as: UTF8.self)
            #expect(first.contains("Profile A"))
            #expect(first.contains("getWeather"))
            #expect(first.contains("0.2"))
            #expect(second.contains("Profile B"))
            #expect(second.contains("otherTool"))
            #expect(second.contains("0.8"))
        }

        func providerProfileCanSwitchModelForToolContinuation(
            _ provider: Provider,
            _ streaming: Bool
        ) async throws {
            UsageURLProtocol.reset()
            if streaming {
                UsageURLProtocol.enqueue(json: try provider.toolStream())
            } else {
                UsageURLProtocol.enqueue(json: try Self.json(provider.response(text: "", tool: true)))
            }
            let state = ProviderProfileState()
            let session = LanguageModelSession(
                profile: ProviderProfile(
                    state: state,
                    model: provider.makeModel(),
                    continuationModel: ProviderProfileTerminalModel(state: state)
                )
            )

            let response =
                try await streaming
                ? session.streamResponse(to: "Weather?").collect()
                : session.respond(to: "Weather?")

            #expect(response.content == "Switched")
            #expect(UsageURLProtocol.recordedBodies.count == 1)
            #expect(response.transcriptEntries.contains { if case .toolCalls = $0 { true } else { false } })
            #expect(response.transcriptEntries.contains { if case .toolOutput = $0 { true } else { false } })
            let selected = try #require(state.selectedRequests.withLock { $0.last })
            #expect(selected.instructions == "Profile B")
            #expect(selected.temperature == 0.8)
            #expect(selected.tools == ["otherTool"])
            #expect(selected.sawToolOutput)
        }

        func providerModelHandoffsShareToolRoundLimit(_ provider: Provider, _ streaming: Bool) async throws {
            UsageURLProtocol.reset()
            let limit = GenerationOptions().maximumToolRounds
            for round in 0 ... limit {
                if streaming {
                    UsageURLProtocol.enqueue(
                        json: try provider.toolStream(
                            city: "City \(round)",
                            callID: "call_\(round)"
                        )
                    )
                } else {
                    let response = try Self.json(provider.response(text: "", tool: true))
                        .replacingOccurrences(of: "Paris", with: "City \(round)")
                        .replacingOccurrences(of: "call_1", with: "call_\(round)")
                    UsageURLProtocol.enqueue(json: response)
                }
            }
            let state = AlternatingProviderProfileState()
            let session = LanguageModelSession(
                profile: AlternatingProviderProfile(
                    state: state,
                    first: provider.makeModel(),
                    second: provider.makeModel()
                )
            )

            let exceeded = await Self.toolRoundLimitExceeded {
                if streaming {
                    _ = try await session.streamResponse(to: "Weather?").collect()
                } else {
                    _ = try await session.respond(to: "Weather?")
                }
            }

            #expect(exceeded?.limit == limit)
            #expect(state.executions.withLock { $0.count } == limit)
            #expect(UsageURLProtocol.recordedBodies.count == limit + 1)
        }

        private enum HandoffResponseFormat: CaseIterable {
            case text, typed, schema, images
        }

        func modelHandoffsRestoreOriginalSchemaPromptOption(_ provider: Provider, _ streaming: Bool) async throws {
            for format in HandoffResponseFormat.allCases {
                for callSiteValue in [false, true] {
                    UsageURLProtocol.reset()
                    UsageURLProtocol.enqueue(
                        json: try streaming
                            ? provider.toolStream()
                            : Self.json(provider.response(text: "", tool: true))
                    )
                    let state = ProviderProfileState()
                    let raw =
                        format == .text
                        ? GeneratedContent("Switched")
                        : GeneratedContent(properties: ["answer": "Switched"])
                    let session = LanguageModelSession(
                        profile: ProviderProfile(
                            state: state,
                            model: provider.makeModel(),
                            continuationModel: ProviderProfileTerminalModel(state: state, rawResponse: raw),
                            initialSchemaOverride: !callSiteValue
                        )
                    )

                    switch format {
                    case .text:
                        let result =
                            try await streaming
                            ? session.streamResponse(
                                to: "Weather?",
                                generating: String.self,
                                includeSchemaInPrompt: callSiteValue
                            ).collect()
                            : session.respond(
                                to: "Weather?",
                                generating: String.self,
                                includeSchemaInPrompt: callSiteValue
                            )
                        #expect(result.content == "Switched")
                    case .typed:
                        let result =
                            try await streaming
                            ? session.streamResponse(
                                to: "Weather?",
                                generating: Answer.self,
                                includeSchemaInPrompt: callSiteValue
                            ).collect()
                            : session.respond(
                                to: "Weather?",
                                generating: Answer.self,
                                includeSchemaInPrompt: callSiteValue
                            )
                        #expect(result.content.answer == "Switched")
                    case .schema:
                        let result =
                            try await streaming
                            ? session.streamResponse(
                                to: "Weather?",
                                schema: Answer.generationSchema,
                                includeSchemaInPrompt: callSiteValue
                            ).collect()
                            : session.respond(
                                to: "Weather?",
                                schema: Answer.generationSchema,
                                includeSchemaInPrompt: callSiteValue
                            )
                        #expect(try result.content.value(String.self, forProperty: "answer") == "Switched")
                    case .images:
                        let result =
                            try await streaming
                            ? session.streamResponse(
                                to: "Weather?",
                                images: [],
                                generating: Answer.self,
                                includeSchemaInPrompt: callSiteValue
                            ).collect()
                            : session.respond(
                                to: "Weather?",
                                images: [],
                                generating: Answer.self,
                                includeSchemaInPrompt: callSiteValue
                            )
                        #expect(result.content.answer == "Switched")
                    }
                    let selected = try #require(state.selectedRequests.withLock { $0.last })
                    #expect(selected.includeSchemaInPrompt == callSiteValue)
                    #expect(UsageURLProtocol.recordedBodies.count == 1)
                }
            }
        }

        func modelHandoffsPreserveEarlierToolResultsAndHistoryWrites(
            _ provider: Provider,
            _ streaming: Bool
        ) async throws {
            UsageURLProtocol.reset()
            for round in 0 ..< 2 {
                let response =
                    try streaming
                    ? provider.toolStream(city: "City \(round)", callID: "call_\(round)")
                    : Self.json(provider.response(text: "", tool: true))
                        .replacingOccurrences(of: "Paris", with: "City \(round)")
                        .replacingOccurrences(of: "call_1", with: "call_\(round)")
                UsageURLProtocol.enqueue(json: response)
            }
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.stream() : Self.json(provider.response())
            )
            let state = AlternatingProviderProfileState(writesHistory: true)
            let session = LanguageModelSession(
                profile: AlternatingProviderProfile(
                    state: state,
                    first: provider.makeModel(),
                    second: provider.makeModel()
                )
            )

            if streaming {
                _ = try await session.streamResponse(to: "Weather?").collect()
            } else {
                _ = try await session.respond(to: "Weather?")
            }

            #expect(state.executions.withLock { $0 } == ["City 0", "City 1"])
            #expect(state.outputHistoryCounts.withLock { $0 } == [1, 2])
            #expect(state.promptIDs.withLock { $0.count } == 1)
            #expect(session.transcript.count == 6)
            #expect(Set(session.transcript.map(\.id)).count == session.transcript.count)
            let lastRequest = try #require(UsageURLProtocol.recordedBodies.last)
            let text = String(decoding: lastRequest, as: UTF8.self)
            #expect(text.contains("Sunny for City 0"))
            #expect(text.contains("Sunny for City 1"))
        }

        func failedHandoffDoesNotPersistPendingHistory(_ provider: Provider, _ streaming: Bool) async throws {
            UsageURLProtocol.reset()
            for round in 0 ..< 2 {
                UsageURLProtocol.enqueue(
                    json: try streaming
                        ? provider.toolStream(city: "City \(round)", callID: "call_\(round)")
                        : Self.json(provider.response(text: "", tool: true))
                            .replacingOccurrences(of: "Paris", with: "City \(round)")
                            .replacingOccurrences(of: "call_1", with: "call_\(round)")
                )
            }
            let state = AlternatingProviderProfileState(writesHistory: true, failsAfterSecondOutput: true)
            let session = LanguageModelSession(
                profile: AlternatingProviderProfile(
                    state: state,
                    first: provider.makeModel(),
                    second: provider.makeModel()
                )
            )

            await #expect(throws: ProfileHistoryError.self) {
                if streaming {
                    _ = try await session.streamResponse(to: "Weather?").collect()
                } else {
                    _ = try await session.respond(to: "Weather?")
                }
            }

            #expect(state.outputHistoryCounts.withLock { $0 } == [1, 2])
            #expect(session.transcript.count == 1)
            #expect(session.transcript.allSatisfy { if case .prompt = $0 { true } else { false } })
        }

        func ordinaryWrapperKeepsTheProviderToolLoop(_ provider: Provider, _ streaming: Bool) async throws {
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.stream() : Self.json(provider.response())
            )
            let invocations = Locked(0)
            let session = LanguageModelSession(
                model: ProviderWrapper(model: provider.makeModel(), invocations: invocations),
                tools: [WeatherTool()]
            )

            let response =
                try await streaming
                ? session.streamResponse(to: "Weather?").collect()
                : session.respond(to: "Weather?")

            #expect(response.content == "Hello")
            #expect(invocations.withLock { $0 } == 1)
            #expect(UsageURLProtocol.recordedBodies.count == 2)
            #expect(session.transcript.count == 4)
        }

        private final class ProviderProfileState: @unchecked Sendable {
            struct SelectedRequest: Sendable {
                let instructions: String?
                let temperature: Double?
                let tools: [String]
                let sawToolOutput: Bool
                let includeSchemaInPrompt: Bool
            }

            let useSecond = Locked(false)
            let executions = Locked<[String]>([])
            let transformSawOutput = Locked<[Bool]>([])
            let events = Locked<[String]>([])
            let selectedRequests = Locked<[SelectedRequest]>([])
        }

        private struct ProviderProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
            let state: ProviderProfileState
            let model: any LanguageModel
            var continuationModel: (any LanguageModel)? = nil
            var initialSchemaOverride: Bool? = nil

            var body: some LanguageModelSession.DynamicProfile {
                if state.useSecond.withLock({ $0 }) {
                    LanguageModelSession.Profile {
                        Instructions("Profile B")
                        ProviderProfileOtherTool()
                    }
                    .model(continuationModel ?? model)
                    .temperature(0.8)
                    .historyTransform(transform)
                    .onToolCall { _ in state.events.withLock { $0.append("call:B") } }
                    .onToolOutput { _, _ in state.events.withLock { $0.append("output:B") } }
                } else {
                    LanguageModelSession.Profile {
                        Instructions("Profile A")
                        ProviderProfileWeatherTool(state: state)
                    }
                    .model(model)
                    .includeSchemaInPrompt(initialSchemaOverride)
                    .temperature(0.2)
                    .historyTransform(transform)
                    .onToolCall { _ in state.events.withLock { $0.append("call:A") } }
                    .onToolOutput { _, _ in state.events.withLock { $0.append("output:A") } }
                }
            }

            private func transform(_ entries: [Transcript.Entry]) -> [Transcript.Entry] {
                state.transformSawOutput.withLock { values in
                    values.append(entries.contains { if case .toolOutput = $0 { true } else { false } })
                }
                return entries
            }
        }

        private final class AlternatingProviderProfileState: @unchecked Sendable {
            let useSecond = Locked(false)
            let executions = Locked<[String]>([])
            let outputHistoryCounts = Locked<[Int]>([])
            let promptIDs = Locked<[String]>([])
            let writesHistory: Bool
            let failsAfterSecondOutput: Bool

            init(writesHistory: Bool = false, failsAfterSecondOutput: Bool = false) {
                self.writesHistory = writesHistory
                self.failsAfterSecondOutput = failsAfterSecondOutput
            }

            func toggleModel() {
                useSecond.withLock { $0.toggle() }
            }
        }

        private struct AlternatingProviderProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
            let state: AlternatingProviderProfileState
            let first: any LanguageModel
            let second: any LanguageModel

            var body: some LanguageModelSession.DynamicProfile {
                if state.useSecond.withLock({ $0 }) {
                    profile(model: second)
                } else {
                    profile(model: first)
                }
            }

            private func profile(model: any LanguageModel) -> some LanguageModelSession.DynamicProfile {
                LanguageModelSession.Profile { AlternatingProviderProfileTool(state: state) }
                    .model(model)
                    .onPrompt { (prompt: Transcript.Prompt) in
                        state.promptIDs.withLock { $0.append(prompt.id) }
                    }
                    .onToolOutput { _, _ in
                        guard state.writesHistory else { return }
                        let properties = try #require(SessionPropertyBinding.values)
                        let history = properties.history
                        let outputs = history.filter { if case .toolOutput = $0 { true } else { false } }
                        state.outputHistoryCounts.withLock { $0.append(outputs.count) }
                        properties.history = history
                        if state.failsAfterSecondOutput && outputs.count == 2 {
                            throw ProfileHistoryError.rejected
                        }
                    }
                    .onResponse { _ in
                        guard state.writesHistory else { return }
                        let properties = try #require(SessionPropertyBinding.values)
                        properties.history = properties.history
                    }
            }
        }

        private struct AlternatingProviderProfileTool: Tool {
            let name = "getWeather"
            let description = "Get weather and switch models"
            let state: AlternatingProviderProfileState

            func call(arguments: WeatherTool.Arguments) async throws -> String {
                state.executions.withLock { $0.append(arguments.city) }
                state.toggleModel()
                return "Sunny for \(arguments.city)"
            }
        }

        private enum ProfileHistoryError: Error {
            case rejected
        }

        private struct ProviderWrapper: LanguageModel {
            typealias UnavailableReason = Never
            let model: any LanguageModel
            let invocations: Locked<Int>

            func respond<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
                invocations.withLock { $0 += 1 }
                return try await model.respond(
                    within: session,
                    to: prompt,
                    generating: type,
                    includeSchemaInPrompt: includeSchemaInPrompt,
                    options: options
                )
            }

            func streamResponse<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
                invocations.withLock { $0 += 1 }
                return model.streamResponse(
                    within: session,
                    to: prompt,
                    generating: type,
                    includeSchemaInPrompt: includeSchemaInPrompt,
                    options: options
                )
            }
        }

        private struct ProviderProfileWeatherTool: Tool {
            let name = "getWeather"
            let description = "Get weather"
            let state: ProviderProfileState

            func call(arguments: WeatherTool.Arguments) async throws -> String {
                state.executions.withLock { $0.append(arguments.city) }
                state.events.withLock { $0.append("execute:A") }
                state.useSecond.withLock { $0 = true }
                return "Sunny"
            }
        }

        struct ProviderProfileOtherTool: Tool {
            @Generable struct Arguments {}
            let name = "otherTool"
            let description = "Only available in profile B"
            func call(arguments: Arguments) async throws -> String { "unused" }
        }

        private struct ProviderProfileTerminalModel: LanguageModel {
            typealias UnavailableReason = Never
            let state: ProviderProfileState
            var rawResponse = GeneratedContent("Switched")

            func respond<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
                let context = session.resolvedRequestContext()
                state.selectedRequests.withLock {
                    $0.append(
                        .init(
                            instructions: context.instructions?.description,
                            temperature: context.options.temperature,
                            tools: context.tools.map(\.name),
                            sawToolOutput: context.transcript.contains {
                                if case .toolOutput = $0 { true } else { false }
                            },
                            includeSchemaInPrompt: includeSchemaInPrompt
                        )
                    )
                }
                let raw = rawResponse
                return .init(
                    content: try Content(raw),
                    rawContent: raw,
                    transcriptEntries: []
                )
            }

            func streamResponse<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
                let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
                    continuation in
                    let task = Task {
                        do {
                            let response: LanguageModelSession.Response<Content> = try await respond(
                                within: session,
                                to: prompt,
                                generating: type,
                                includeSchemaInPrompt: includeSchemaInPrompt,
                                options: options
                            )
                            continuation.yield(
                                .init(
                                    content: response.content.asPartiallyGenerated(),
                                    rawContent: response.rawContent
                                )
                            )
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }
                    continuation.onTermination = { _ in task.cancel() }
                }
                return .init(stream: stream)
            }
        }
    }
#endif
