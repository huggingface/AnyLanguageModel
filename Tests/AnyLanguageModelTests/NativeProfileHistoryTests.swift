import Foundation
import Testing

#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS) && !os(watchOS)
    import FoundationModels
    @_spi(Compatibility) @testable import AnyLanguageModel

    @Suite("Native profile history projection")
    struct NativeProfileHistoryTests {
        @Test func nativeSessionUsesProjectedHistoryWithoutMutatingCanonicalHistory() throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let earlier = AnyLanguageModel.Transcript.Entry.prompt(
                .init(segments: [.text(.init(content: "Earlier"))])
            )
            let prompt = AnyLanguageModel.Prompt("Current")
            let current = AnyLanguageModel.Transcript.Entry.prompt(
                .init(id: "current", segments: try prompt.makeTranscriptSegments(occurrenceID: "current"))
            )
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(MockLanguageModel())
                .historyTransform { $0.filter { $0.id != earlier.id } },
                history: [earlier, current]
            )
            let native = try makeFoundationModelsRequest(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: prompt,
                fmPrompt: prompt.toFoundationModels()
            ).session
            #expect(!native.transcript.contains { if case .prompt = $0 { true } else { false } })
            let canonical = session.transcript.filter { entry in
                if case .instructions = entry { return false }
                return true
            }
            #expect(canonical == [earlier, current])
        }

        @Test(arguments: PromptRewrite.allCases, [false, true])
        fileprivate func rewrittenCurrentPromptIsTheRequestsOnlyCopyOfThatPrompt(
            rewrite: PromptRewrite,
            streaming: Bool
        ) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let earlier = AnyLanguageModel.Transcript.Entry.prompt(
                .init(segments: [.text(.init(content: "Earlier"))])
            )
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { $0.map { rewrite($0) } },
                history: [earlier]
            )

            try await run(session, streaming: streaming)

            // Foundation Models sends the transform's output and keeps the prompts in its transcript.
            #expect(recording.requestTexts == [["Rewritten Earlier", "Rewritten Current"]])
            #expect(promptTexts(session.transcript) == ["Earlier", "Current"])
        }

        @Test(arguments: [false, true])
        func unchangedCurrentPromptIsSentOnce(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let earlier = AnyLanguageModel.Transcript.Entry.prompt(
                .init(segments: [.text(.init(content: "Earlier"))])
            )
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { $0.filter { $0.id != earlier.id } },
                history: [earlier]
            )

            try await run(session, streaming: streaming)

            #expect(recording.requestTexts == [["Current"]])
            #expect(promptTexts(session.transcript) == ["Earlier", "Current"])
        }

        @Test(arguments: [false, true])
        func rewrittenPromptReplacesARemoteImagePrompt(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform(replacingPrompts(with: [.text(.init(content: "Rewritten"))]))
            )

            try await run(session, prompt: try remoteImagePrompt(), streaming: streaming)

            #expect(recording.requestTexts == [["Rewritten"]])
        }

        @Test(arguments: [false, true])
        func profileKeepingARemoteImagePromptIsRejected(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
            )
            let prompt = try remoteImagePrompt()

            await #expect(throws: AnyLanguageModel.PromptBridgeError.self) {
                try await run(session, prompt: prompt, streaming: streaming)
            }
            #expect(recording.requestTexts.isEmpty)
        }

        @Test(arguments: [false, true])
        func rewritingToARemoteImageIsRejected(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let url = try #require(URL(string: "https://example.com/image.png"))
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform(replacingPrompts(with: [.image(.init(url: url))]))
            )

            await #expect(throws: AnyLanguageModel.PromptBridgeError.self) {
                try await run(session, streaming: streaming)
            }
            #expect(recording.requestTexts.isEmpty)
        }

        /// Plain sessions and ordinary dynamic instructions reject the prompt
        /// before they load the model.
        @Test(arguments: [false, true], [false, true])
        func sessionsWithoutProfilesRejectARemoteImageBeforeLoadingTheModel(
            dynamic: Bool,
            streaming: Bool
        ) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let loads = Locked(0)
            let model = AnyLanguageModel.FoundationLanguageModel<RecordingModel>(loading: {
                loads.withLock { $0 += 1 }
                return RecordingModel()
            })
            let session =
                dynamic
                ? AnyLanguageModel.LanguageModelSession(
                    model: model,
                    dynamicInstructions: CountingProjectionInstructions(evaluations: Locked(0))
                )
                : AnyLanguageModel.LanguageModelSession(model: model)
            let prompt = try remoteImagePrompt()

            await #expect(throws: AnyLanguageModel.PromptBridgeError.self) {
                try await run(session, prompt: prompt, streaming: streaming)
            }
            #expect(loads.withLock { $0 } == 0)
        }

        @Test(arguments: [false, true])
        fileprivate func rewrittenPromptKeepsStructuredSegments(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let (segments, recorded) = try structuredSegments()
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform(replacingPrompts(with: segments))
            )

            try await run(session, streaming: streaming)

            #expect(recording.requests.withLock { $0 } == [[recorded]])
            #expect(promptTexts(session.transcript) == ["Current"])
        }

        @Test(arguments: [false, true])
        fileprivate func rewrittenPromptStaysThroughAToolRound(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let (segments, recorded) = try structuredSegments()
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform(replacingPrompts(with: segments))
            )

            try await run(session, streaming: streaming)

            #expect(recording.requests.withLock { $0 } == [[recorded], [recorded]])
        }

        @Test(arguments: [false, true])
        func nativeRequestEvaluatesTheProfileOnce(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    CountingProjectionInstructions(evaluations: evaluations)
                }
                .model(AnyLanguageModel.FoundationLanguageModel(RecordingModel()))
            )
            // Creating a profile session evaluates the profile once to record its instructions.
            #expect(evaluations.withLock { $0 } == 1)

            try await run(session, streaming: streaming)

            #expect(evaluations.withLock { $0 } == 2)
        }

        @Test func ordinaryDynamicInstructionsAreEvaluatedOnlyByTheNativeSession() throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                model: MockLanguageModel(),
                dynamicInstructions: CountingProjectionInstructions(evaluations: evaluations)
            )
            _ = try makeFoundationModelsRequest(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: AnyLanguageModel.Prompt("Current"),
                fmPrompt: FoundationModels.Prompt("Current")
            )
            #expect(evaluations.withLock { $0 } == 1)
            _ = FoundationModelsDynamicInstructionsAdapter(session: session).body
            #expect(evaluations.withLock { $0 } == 2)
        }
    }

    /// How a history transform rewrites a prompt, which decides the IDs that its entry keeps.
    private enum PromptRewrite: CaseIterable, Sendable {
        case newEntry
        case sameEntryID
        case sameSegmentIDs

        func callAsFunction(_ entry: AnyLanguageModel.Transcript.Entry) -> AnyLanguageModel.Transcript.Entry {
            guard case .prompt(let prompt) = entry else { return entry }
            let text = "Rewritten \(promptText(prompt.segments))"
            switch self {
            case .newEntry:
                return .prompt(.init(segments: [.text(.init(content: text))]))
            case .sameEntryID:
                return .prompt(.init(id: prompt.id, segments: [.text(.init(content: text))]))
            case .sameSegmentIDs:
                let segments = prompt.segments.map { segment -> AnyLanguageModel.Transcript.Segment in
                    .text(.init(id: segment.id, content: text))
                }
                return .prompt(.init(id: prompt.id, segments: segments))
            }
        }
    }

    private func promptText(_ segments: [AnyLanguageModel.Transcript.Segment]) -> String {
        segments.map { segment in
            if case .text(let text) = segment { return text.content }
            return "?"
        }.joined()
    }

    private func promptTexts(_ entries: some Sequence<AnyLanguageModel.Transcript.Entry>) -> [String] {
        entries.compactMap { entry in
            guard case .prompt(let prompt) = entry else { return nil }
            return promptText(prompt.segments)
        }
    }

    @available(macOS 27, iOS 27, visionOS 27, *)
    private func run(
        _ session: AnyLanguageModel.LanguageModelSession,
        prompt: AnyLanguageModel.Prompt = AnyLanguageModel.Prompt("Current"),
        streaming: Bool
    ) async throws {
        if streaming {
            for try await _ in session.streamResponse(to: prompt) {}
        } else {
            _ = try await session.respond(to: prompt)
        }
    }

    private func remoteImagePrompt() throws -> AnyLanguageModel.Prompt {
        let url = try #require(URL(string: "https://example.com/image.png"))
        return AnyLanguageModel.Prompt { AnyLanguageModel.Attachment(imageURL: url) }
    }

    /// Replaces each prompt in a history with one that has the given segments.
    private func replacingPrompts(
        with segments: [AnyLanguageModel.Transcript.Segment]
    ) -> ([AnyLanguageModel.Transcript.Entry]) -> [AnyLanguageModel.Transcript.Entry] {
        { entries in
            entries.map { entry in
                guard case .prompt = entry else { return entry }
                return .prompt(.init(segments: segments))
            }
        }
    }

    private func structuredSegments() throws -> (
        segments: [AnyLanguageModel.Transcript.Segment],
        recorded: [RecordedSegment]
    ) {
        let json = #"{"name":"Rex"}"#
        let segments: [AnyLanguageModel.Transcript.Segment] = [
            .text(.init(id: "text-segment", content: "See")),
            .structure(
                .init(
                    id: "structure-segment",
                    source: "Pet",
                    content: try AnyLanguageModel.GeneratedContent(json: json)
                )
            ),
        ]
        guard #available(macOS 27, iOS 27, visionOS 27, *) else { return (segments, []) }
        return (
            segments,
            [
                .text("See"),
                .structure(
                    id: "structure-segment",
                    source: "Pet",
                    content: try FoundationModels.GeneratedContent(json: json).jsonString
                ),
            ]
        )
    }

    /// A segment of a prompt that Foundation Models sends to a model.
    private enum RecordedSegment: Equatable, Sendable {
        case text(String)
        case structure(id: String, source: String, content: String)
        case image
        case other
    }

    /// A model that records the prompts of each request that Foundation Models sends it.
    @available(macOS 27, iOS 27, visionOS 27, *)
    private struct RecordingModel: FoundationModels.LanguageModel {
        typealias Executor = RecordingExecutor
        let requests = Locked<[[[RecordedSegment]]]>([])
        var callsTool = false
        var capabilities: FoundationModels.LanguageModelCapabilities { .init([.vision, .toolCalling]) }
        var executorConfiguration: String { "native-profile-history" }

        var requestTexts: [[String]] {
            requests.withLock { requests in
                requests.map { prompts in
                    prompts.map { segments in
                        segments.map { segment in
                            if case .text(let text) = segment { return text }
                            return "?"
                        }.joined()
                    }
                }
            }
        }
    }

    @available(macOS 27, iOS 27, visionOS 27, *)
    private struct RecordingExecutor: FoundationModels.LanguageModelExecutor {
        init(configuration: String) {}

        func respond(
            to request: FoundationModels.LanguageModelExecutorGenerationRequest,
            model: RecordingModel,
            streamingInto channel: FoundationModels.LanguageModelExecutorGenerationChannel
        ) async throws {
            let prompts = request.transcript.compactMap { entry -> [RecordedSegment]? in
                guard case .prompt(let prompt) = entry else { return nil }
                return prompt.segments.map { segment in
                    switch segment {
                    case .text(let text):
                        return .text(text.content)
                    case .structure(let structure):
                        return .structure(
                            id: structure.id,
                            source: structure.schemaName,
                            content: structure.content.jsonString
                        )
                    case .attachment:
                        return .image
                    @unknown default:
                        return .other
                    }
                }
            }
            model.requests.withLock { $0.append(prompts) }
            let hasToolOutput = request.transcript.contains { entry in
                if case .toolOutput = entry { return true }
                return false
            }
            if model.callsTool, !hasToolOutput {
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: "weather-call",
                            name: "getWeather",
                            action: .appendArguments(#"{"city":"Paris"}"#, tokenCount: 1)
                        )
                    )
                )
                return
            }
            await channel.send(.response(action: .appendText("OK", tokenCount: 1)))
        }
    }

    private struct WeatherInstructions: AnyLanguageModel.DynamicInstructions {
        var body: some AnyLanguageModel.DynamicInstructions {
            AnyLanguageModel.Instructions("Projected")
            WeatherTool()
        }
    }

    private struct CountingProjectionInstructions: AnyLanguageModel.DynamicInstructions {
        let evaluations: Locked<Int>

        var body: some AnyLanguageModel.DynamicInstructions {
            let count = evaluations.withLock {
                $0 += 1; return $0
            }
            AnyLanguageModel.Instructions("Revision \(count)")
        }
    }
#endif
