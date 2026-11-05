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
            #expect(recording.requests.withLock { $0 } == [["Rewritten Earlier", "Rewritten Current"]])
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

            #expect(recording.requests.withLock { $0 } == [["Current"]])
            #expect(promptTexts(session.transcript) == ["Earlier", "Current"])
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

        @Test func nativeProfileConstructionUsesOneContextForHistoryAndInstructions() throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    CountingProjectionInstructions(evaluations: evaluations)
                }
                .model(MockLanguageModel())
            )
            // Creating a profile session evaluates the profile once to record its instructions.
            #expect(evaluations.withLock { $0 } == 1)
            _ = try makeFoundationModelsRequest(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: AnyLanguageModel.Prompt("Current"),
                fmPrompt: FoundationModels.Prompt("Current")
            )
            #expect(evaluations.withLock { $0 } == 2)
        }

        @Test func nativeProfileInitialContextSurvivesDetachedEvaluationWithoutFreezingLaterRequests() async {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    CountingProjectionInstructions(evaluations: evaluations)
                }
                .model(MockLanguageModel())
            )
            // Creating a profile session evaluates the profile once to record its instructions.
            #expect(evaluations.withLock { $0 } == 1)
            let initialContext = session.resolvedRequestContext()
            #expect(evaluations.withLock { $0 } == 2)
            let adapter = FoundationModelsDynamicInstructionsAdapter(session: session, initialContext: initialContext)
            await Task.detached {
                #expect(AnyLanguageModel.SessionPropertyBinding.values == nil)
                _ = adapter.body
            }.value
            #expect(evaluations.withLock { $0 } == 2)

            await Task.detached { _ = adapter.body }.value
            #expect(evaluations.withLock { $0 } == 3)
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
    private func run(_ session: AnyLanguageModel.LanguageModelSession, streaming: Bool) async throws {
        if streaming {
            for try await _ in session.streamResponse(to: "Current") {}
        } else {
            _ = try await session.respond(to: "Current")
        }
    }

    /// A model that records the prompts of each request that Foundation Models sends it.
    @available(macOS 27, iOS 27, visionOS 27, *)
    private struct RecordingModel: FoundationModels.LanguageModel {
        typealias Executor = RecordingExecutor
        let requests = Locked<[[String]]>([])
        var capabilities: FoundationModels.LanguageModelCapabilities { .init([.toolCalling]) }
        var executorConfiguration: String { "native-profile-history" }
    }

    @available(macOS 27, iOS 27, visionOS 27, *)
    private struct RecordingExecutor: FoundationModels.LanguageModelExecutor {
        init(configuration: String) {}

        func respond(
            to request: FoundationModels.LanguageModelExecutorGenerationRequest,
            model: RecordingModel,
            streamingInto channel: FoundationModels.LanguageModelExecutorGenerationChannel
        ) async throws {
            let prompts = request.transcript.compactMap { entry -> String? in
                guard case .prompt(let prompt) = entry else { return nil }
                return prompt.segments.map { segment in
                    if case .text(let text) = segment { return text.content }
                    return "?"
                }.joined()
            }
            model.requests.withLock { $0.append(prompts) }
            await channel.send(.response(action: .appendText("OK", tokenCount: 1)))
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
