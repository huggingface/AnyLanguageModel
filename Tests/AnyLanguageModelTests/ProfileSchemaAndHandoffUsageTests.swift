import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(Darwin) && !canImport(AsyncHTTPClient)
    extension ProviderUsageTests {
        /// Enqueues a tool round and a structured answer.
        private func enqueueToolRoundThenAnswer(_ provider: Provider, streaming: Bool) throws {
            let final = #"{"answer":"OK"}"#
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.toolStream() : Self.json(provider.response(text: "", tool: true))
            )
            UsageURLProtocol.enqueue(
                json: try streaming ? provider.stream(text: final) : Self.json(provider.response(text: final))
            )
        }

        /// How many times each recorded request says the schema in words.
        private func schemaTextCounts() -> [Int] {
            UsageURLProtocol.recordedBodies.map { body in
                String(decoding: body, as: UTF8.self)
                    .components(separatedBy: "Respond with valid JSON").count - 1
            }
        }

        private func respondWithAnswer(
            _ session: LanguageModelSession,
            streaming: Bool
        ) async throws -> Answer {
            try await streaming
                ? session.streamResponse(to: "Weather?", generating: Answer.self).collect().content
                : session.respond(to: "Weather?", generating: Answer.self).content
        }

        @Test(arguments: Provider.allCases, [false, true])
        func profileSaysSchemaOnceInEveryRequest(_ provider: Provider, _ streaming: Bool) async throws {
            try enqueueToolRoundThenAnswer(provider, streaming: streaming)
            let session = LanguageModelSession(
                profile: LanguageModelSession.Profile { WeatherTool() }.model(provider.makeModel())
            )

            let answer = try await respondWithAnswer(session, streaming: streaming)

            #expect(answer.answer == "OK")
            #expect(schemaTextCounts() == [1, 1])
        }

        @Test(arguments: Provider.allCases, [false, true])
        func schemaSettingChangeAppliesToTheNextRequest(_ provider: Provider, _ streaming: Bool) async throws {
            for initial in [false, true] {
                try enqueueToolRoundThenAnswer(provider, streaming: streaming)
                let setting = Locked(initial)
                let session = LanguageModelSession(
                    profile: SchemaSettingProfile(setting: setting, model: provider.makeModel())
                )

                let answer = try await respondWithAnswer(session, streaming: streaming)

                #expect(answer.answer == "OK")
                #expect(schemaTextCounts() == [initial ? 1 : 0, initial ? 0 : 1])
            }
        }

        @Test(arguments: Provider.allCases)
        func streamingHandoffCountsPrefixAndPrepaidRequestsOnce(_ provider: Provider) async throws {
            let earlier = LanguageModelSession.Usage(
                input: .init(totalTokenCount: 7, cachedTokenCount: 1),
                output: .init(totalTokenCount: 3, reasoningTokenCount: 0)
            )
            let final = LanguageModelSession.Usage(
                input: .init(totalTokenCount: 40, cachedTokenCount: 2),
                output: .init(totalTokenCount: 11, reasoningTokenCount: 0)
            )
            let requests = RequestsUsage(earlier: earlier, final: final)
            // The second case reports the same usage as the prefix.
            for local in [requests, RequestsUsage(final: provider.expected)] {
                UsageURLProtocol.reset()
                UsageURLProtocol.enqueue(json: try provider.toolStream())
                let stage = Locked(0)
                let session = LanguageModelSession(
                    profile: StagedModelProfile(
                        stage: stage,
                        models: [provider.makeModel(), PrepayingStructuredModel(requests: local)]
                    )
                )

                let response = try await session.streamResponse(to: "Weather?", generating: Answer.self).collect()

                #expect(response.content.answer == "Local")
                var expected = provider.expected
                expected.add(local.total)
                #expect(session.usage == expected)
                #expect(UsageURLProtocol.recordedBodies.count == 1)
            }
        }

        @Test(arguments: Provider.allCases)
        func nestedStreamingHandoffsCountEveryPrefixOnce(_ provider: Provider) async throws {
            let local = RequestsUsage(
                final: .init(
                    input: .init(totalTokenCount: 40, cachedTokenCount: 2),
                    output: .init(totalTokenCount: 11, reasoningTokenCount: 0)
                )
            )
            UsageURLProtocol.reset()
            UsageURLProtocol.enqueue(json: try provider.toolStream())
            UsageURLProtocol.enqueue(json: try provider.toolStream(callID: "call_2"))
            let stage = Locked(0)
            let session = LanguageModelSession(
                profile: StagedModelProfile(
                    stage: stage,
                    models: [
                        provider.makeModel(),
                        provider.makeModel(),
                        PrepayingStructuredModel(requests: local),
                    ]
                )
            )

            let response = try await session.streamResponse(to: "Weather?", generating: Answer.self).collect()

            #expect(response.content.answer == "Local")
            var expected = provider.expected
            expected.add(provider.expected)
            expected.add(local.total)
            #expect(session.usage == expected)
            #expect(UsageURLProtocol.recordedBodies.count == 2)
        }

        @Test func streamingHandoffCountsAPrefixThatNoSnapshotReported() async throws {
            let prefix = LanguageModelSession.Usage(
                input: .init(totalTokenCount: 100, cachedTokenCount: 0),
                output: .init(totalTokenCount: 20, reasoningTokenCount: 0)
            )
            let local = RequestsUsage(
                final: .init(
                    input: .init(totalTokenCount: 40, cachedTokenCount: 0),
                    output: .init(totalTokenCount: 11, reasoningTokenCount: 0)
                )
            )
            let stage = Locked(0)
            let session = LanguageModelSession(
                profile: StagedModelProfile(
                    stage: stage,
                    models: [
                        SilentPrefixModel(stage: stage, prefixUsage: prefix),
                        PrepayingStructuredModel(requests: local),
                    ]
                )
            )

            let response = try await session.streamResponse(to: "Weather?", generating: Answer.self).collect()

            #expect(response.content.answer == "Local")
            var expected = prefix
            expected.add(local.total)
            #expect(session.usage == expected)
        }

        @Test func prepaidSnapshotCountsOnlyTheUncountedPrefix() {
            let prefix = LanguageModelSession.Usage(
                input: .init(totalTokenCount: 100, cachedTokenCount: 0),
                output: .init(totalTokenCount: 20, reasoningTokenCount: 0)
            )
            let local = LanguageModelSession.Usage(
                input: .init(totalTokenCount: 40, cachedTokenCount: 0),
                output: .init(totalTokenCount: 11, reasoningTokenCount: 0)
            )
            var total = prefix
            total.add(local)
            for alreadyCounted in [LanguageModelSession.Usage.zero, prefix] {
                let recorded = Locked(LanguageModelSession.Usage.zero)
                let account = CountedRequestsUsage { usage in recorded.withLock { $0.add(usage) } }
                account.count(RequestsUsage(final: alreadyCounted), prepaid: nil)
                let prepaid = account.prepay(RequestsUsage(final: local))
                account.count(
                    RequestsUsage(earlier: prefix, final: local),
                    prepaid: prepaid,
                    unprepaidUsage: prefix
                )
                #expect(recorded.withLock { $0 } == total)
            }
        }

        private struct SchemaSettingProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
            let setting: Locked<Bool>
            let model: any LanguageModel

            var body: some LanguageModelSession.DynamicProfile {
                LanguageModelSession.Profile { SchemaSettingTool(setting: setting) }
                    .model(model)
                    .includeSchemaInPrompt(setting.withLock { $0 })
            }
        }

        private struct SchemaSettingTool: Tool {
            let name = "getWeather"
            let description = "Get weather and change the schema setting"
            let setting: Locked<Bool>

            func call(arguments: WeatherTool.Arguments) async throws -> String {
                setting.withLock { $0.toggle() }
                return "Sunny"
            }
        }

        /// Selects the next model after each tool call.
        private struct StagedModelProfile: LanguageModelSession.DynamicProfile, @unchecked Sendable {
            let stage: Locked<Int>
            let models: [any LanguageModel]

            var body: some LanguageModelSession.DynamicProfile {
                LanguageModelSession.Profile { StagedModelTool(stage: stage) }
                    .model(models[stage.withLock { $0 }])
            }
        }

        private struct StagedModelTool: Tool {
            let name = "getWeather"
            let description = "Get weather and select the next model"
            let stage: Locked<Int>

            func call(arguments: WeatherTool.Arguments) async throws -> String {
                stage.withLock { $0 += 1 }
                return "Sunny"
            }
        }

        /// A model that hands off after a request whose usage no snapshot reports.
        private struct SilentPrefixModel: LanguageModel, ProfileRequestIdentifiedLanguageModel {
            typealias UnavailableReason = Never
            let profileRequestIdentity = UUID()
            let stage: Locked<Int>
            let prefixUsage: LanguageModelSession.Usage

            func respond<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
                try await streamResponse(
                    within: session,
                    to: prompt,
                    generating: type,
                    includeSchemaInPrompt: includeSchemaInPrompt,
                    options: options
                ).collect()
            }

            func streamResponse<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
                let prefixEntries: [Transcript.Entry] = [
                    .response(.init(assetIDs: [], segments: [.text(.init(content: "Checking"))]))
                ]
                let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
                    continuation in
                    let task = Task {
                        do {
                            stage.withLock { $0 = 1 }
                            let context = try await session.resolvedRequestContext(
                                including: prefixEntries,
                                options: options
                            )
                            let handoff = try #require(
                                session.profileContinuationStream(
                                    ifSelectedModelDiffersFrom: profileRequestIdentity,
                                    requestContext: context,
                                    prompt: prompt,
                                    generating: type,
                                    schema: type.generationSchema,
                                    includeSchemaInPrompt: includeSchemaInPrompt,
                                    prefixEntries: prefixEntries,
                                    prefixUsage: prefixUsage,
                                    prefixText: ""
                                )
                            )
                            for try await snapshot in handoff { continuation.yield(snapshot) }
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

        /// A local structured model, which prepays its requests before it yields its one snapshot.
        private struct PrepayingStructuredModel: LanguageModel {
            typealias UnavailableReason = Never
            let requests: RequestsUsage

            func respond<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
                let raw = GeneratedContent(properties: ["answer": "Local"])
                return .init(
                    content: try Content(raw),
                    rawContent: raw,
                    transcriptEntries: [],
                    requests: requests
                )
            }

            func streamResponse<Content>(
                within session: LanguageModelSession,
                to prompt: Prompt,
                generating type: Content.Type,
                includeSchemaInPrompt: Bool,
                options: GenerationOptions
            ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
                streamStructuredResponse(
                    within: session,
                    to: prompt,
                    generating: type,
                    includeSchemaInPrompt: includeSchemaInPrompt,
                    options: options
                )
            }
        }
    }
#endif
