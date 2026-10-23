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
            let native = makeFoundationModelsSession(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: prompt
            )
            #expect(!native.transcript.contains { if case .prompt = $0 { true } else { false } })
            #expect(Array(session.transcript) == [earlier, current])
        }

        @Test func ordinaryDynamicInstructionsAreEvaluatedOnlyByTheNativeSession() {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                model: MockLanguageModel(),
                dynamicInstructions: CountingProjectionInstructions(evaluations: evaluations)
            )
            _ = makeFoundationModelsSession(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: AnyLanguageModel.Prompt("Current")
            )
            #expect(evaluations.withLock { $0 } == 1)
            _ = FoundationModelsDynamicInstructionsAdapter(session: session).body
            #expect(evaluations.withLock { $0 } == 2)
        }

        @Test func nativeProfileConstructionUsesOneContextForHistoryAndInstructions() {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    CountingProjectionInstructions(evaluations: evaluations)
                }
                .model(MockLanguageModel())
            )
            _ = makeFoundationModelsSession(
                model: FoundationModels.SystemLanguageModel.default,
                session: session,
                prompt: AnyLanguageModel.Prompt("Current")
            )
            #expect(evaluations.withLock { $0 } == 1)
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
            let initialContext = session.resolvedRequestContext()
            let adapter = FoundationModelsDynamicInstructionsAdapter(session: session, initialContext: initialContext)
            await Task.detached {
                #expect(AnyLanguageModel.SessionPropertyBinding.values == nil)
                _ = adapter.body
            }.value
            #expect(evaluations.withLock { $0 } == 1)

            await Task.detached { _ = adapter.body }.value
            #expect(evaluations.withLock { $0 } == 2)
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
