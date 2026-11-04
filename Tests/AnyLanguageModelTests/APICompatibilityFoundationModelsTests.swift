import Testing

#if canImport(FoundationModels) && !os(watchOS)
    import FoundationModels

    private let isFoundationModelsSystemLanguageModelAvailable: Bool = {
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return SystemLanguageModel.default.isAvailable
        }
        return false
    }()

    #if compiler(>=6.4)
        #if os(macOS) || os(iOS) || os(visionOS)
            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            @Test func foundationModelsHistoryViewCollectionCompatibility() {
                func requireContract<C>(_ history: C)
                where
                    C: RandomAccessCollection & MutableCollection & RangeReplaceableCollection
                        & ExpressibleByArrayLiteral & Sendable,
                    C.Element == Transcript.Entry, C.SubSequence == C, C.Index: Hashable & Strideable,
                    C.Index.Stride == Int
                {}
                var history: Transcript.HistoryView = []
                let empty: Transcript.HistoryView = history[history.startIndex ..< history.endIndex]
                history[history.startIndex ..< history.endIndex] = empty
                history.append(contentsOf: empty)
                history.replaceSubrange(history.startIndex ..< history.endIndex, with: empty)
                requireContract(history)
                #expect(history.isEmpty)
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            extension SessionPropertyValues {
                @SessionPropertyEntry var compatibilityRevision = 0
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private struct CompatibilityDynamicInstructions: DynamicInstructions {
                let includeDetail: Bool
                @SessionProperty(\.compatibilityRevision) private var revision

                var body: some DynamicInstructions {
                    Instructions("You are a helpful assistant. Revision \(revision).")
                    if includeDetail {
                        Instructions("Include useful detail.")
                    }
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private struct CompatibilityDynamicProfile: LanguageModelSession.DynamicProfile {
                let model: any LanguageModel
                @SessionProperty(\.compatibilityRevision) private var revision

                var body: some LanguageModelSession.DynamicProfile {
                    Profile {
                        Instructions("Revision \(revision)")
                    }
                    .model(model)
                    .samplingMode(.greedy)
                    .temperature(0.2)
                    .maximumResponseTokens(32)
                    .reasoningLevel(.light)
                    .toolCallingMode(.allowed)
                    .historyTransform { $0 }
                    .transcriptErrorHandlingPolicy(nil)
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            @Test func foundationModelsNonSendableHistoryTransformCompatibility() {
                let captured = NonSendableHistoryCapture()
                let transform: ([Transcript.Entry]) -> [Transcript.Entry] = { entries in
                    captured.calls += 1
                    return entries
                }
                _ = LanguageModelSession.AnyDynamicProfile(
                    LanguageModelSession.Profile { Instructions("Projection") }
                        .historyTransform(transform)
                )
                #expect(captured.calls == 0)
            }

            private final class NonSendableHistoryCapture {
                var calls = 0
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private struct CompatibilityCallbackProfile: LanguageModelSession.DynamicProfile {
                let model: any LanguageModel

                var body: some LanguageModelSession.DynamicProfile {
                    Profile {
                        Instructions("Callbacks")
                    }
                    .model(model)
                    .onPrompt { (_: Transcript.Prompt) in }
                    .onPrompt {}
                    .onResponse { (_: Transcript.Response) in }
                    .onResponse {}
                    .onReasoning { (_: Transcript.Reasoning) in }
                    .onReasoning {}
                    .onToolCall { (_: Transcript.ToolCall) in }
                    .onToolCall {}
                    .onToolOutput { (_: Transcript.ToolCall, _: Transcript.ToolOutput) in }
                    .onToolOutput {}
                    .onActivate {}
                    .onDeactivate {}
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private struct CompatibilityProfileModifier: LanguageModelSession.DynamicProfileModifier {
                let temperature: Double

                func body(content: Content) -> some DynamicProfile {
                    content
                        .temperature(temperature)
                        .onResponse {}
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private struct CompatibilityConditionalProfile: LanguageModelSession.DynamicProfile {
                let model: any LanguageModel
                let includeDetail: Bool

                var body: some LanguageModelSession.DynamicProfile {
                    if includeDetail {
                        Profile {
                            Instructions("Include useful detail.")
                        }
                        .model(model)
                    } else {
                        Profile {
                            Instructions("Be brief.")
                        }
                        .model(model)
                        .modifier(CompatibilityProfileModifier(temperature: 0.3))
                    }
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private struct CompatibilityAvailabilityProfile: LanguageModelSession.DynamicProfile {
                let model: any LanguageModel

                var body: some LanguageModelSession.DynamicProfile {
                    if #available(macOS 28.0, iOS 28.0, visionOS 28.0, *) {
                        Profile {
                            Instructions("Later releases")
                        }
                        .model(model)
                    } else {
                        Profile {
                            Instructions("Earlier releases")
                        }
                        .model(model)
                    }
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            @Test func foundationModelsDynamicProfileCompatibility() {
                let model = SystemLanguageModel.default
                let session = LanguageModelSession(
                    profile: CompatibilityCallbackProfile(model: model),
                    history: [Transcript.Entry]()
                )
                _ = LanguageModelSession(profile: CompatibilityConditionalProfile(model: model, includeDetail: true))
                _ = LanguageModelSession(profile: CompatibilityConditionalProfile(model: model, includeDetail: false))
                _ = LanguageModelSession(profile: CompatibilityAvailabilityProfile(model: model))
                _ = LanguageModelSession.AnyDynamicProfile(
                    LanguageModelSession.Profile {
                        Instructions("Custom modifier")
                    }
                    .model(model)
                    .modifier(CompatibilityProfileModifier(temperature: 0.5))
                )
                // Creating a profile session records the profile's instructions in the transcript.
                #expect(session.transcript.count == 1)
                #expect(foundationModelsInstructionTexts(session.transcript) == [["Callbacks"]])
                #expect(session.properties.history.isEmpty)

                // A leading instructions entry in the history keeps its ID
                // and takes the active profile's instructions.
                let base = Transcript.Instructions(segments: [.text(.init(content: "Base"))], toolDefinitions: [])
                let restored = LanguageModelSession(
                    profile: CompatibilityConditionalProfile(model: model, includeDetail: false),
                    history: [Transcript.Entry.instructions(base)]
                )
                #expect(restored.transcript.map(\.id) == [base.id])
                #expect(foundationModelsInstructionTexts(restored.transcript) == [["Be brief."]])
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            private func foundationModelsInstructionTexts(_ transcript: Transcript) -> [[String]] {
                transcript.compactMap { entry -> [String]? in
                    guard case .instructions(let instructions) = entry else { return nil }
                    return instructions.segments.compactMap { segment -> String? in
                        if case .text(let text) = segment { return text.content }
                        return nil
                    }
                }
            }

            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            @Test func foundationModelsNonSendableProfileCallbackCompatibility() {
                // The callback takes the non-Sendable state that it captures,
                // so this function can't use it again.
                let captured = NonSendableCallbackCapture()
                _ = LanguageModelSession.AnyDynamicProfile(
                    LanguageModelSession.Profile { Instructions("Callbacks") }
                        .model(SystemLanguageModel.default)
                        .onResponse { captured.calls += 1 }
                )
            }

            private final class NonSendableCallbackCapture {
                var calls = 0
            }
        #endif
    #endif

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test(
        "FoundationModels Drop-In Compatibility",
        .enabled(if: isFoundationModelsSystemLanguageModelAvailable)
    )
    func foundationModelsCompatibility() async throws {
        let model = SystemLanguageModel.default
        let session = LanguageModelSession(
            model: model,
            instructions: Instructions("You are a helpful assistant.")
        )

        #if compiler(>=6.4)
            #if os(macOS) || os(iOS) || os(visionOS)
                if #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) {
                    let dynamicSession = LanguageModelSession(
                        model: model,
                        dynamicInstructions: CompatibilityDynamicInstructions(includeDetail: true),
                        history: session.transcript
                    )
                    dynamicSession.properties.compatibilityRevision = 1
                    _ = dynamicSession.properties.history
                    _ = LanguageModelSession.AnyDynamicProfile(CompatibilityDynamicProfile(model: model))
                }
            #endif
        #endif

        let options = GenerationOptions(temperature: 0.7)
        let response = try await session.respond(options: options) {
            Prompt("Say 'Hello'")
        }
        #expect(!response.content.isEmpty)

        let stream = session.streamResponse {
            Prompt("Count to 3")
        }
        var hasSnapshots = false
        for try await _ in stream {
            hasSnapshots = true
            break
        }
        #expect(hasSnapshots)
    }

    #if compiler(>=6.4)
        #if os(macOS) || os(iOS) || os(visionOS)
            @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
            @Test("FoundationModels request options compatibility")
            func foundationModelsRequestOptionsCompatibility() {
                var options = GenerationOptions(samplingMode: .greedy)
                options.samplingMode = .random(top: 40, seed: 1)
                options.toolCallingMode = .required
                #expect(options.samplingMode == .random(top: 40, seed: 1))
                #expect(options.toolCallingMode?.kind == .required)
                #expect(GenerationOptions(temperature: 0.5).toolCallingMode == nil)
                let disallowed = GenerationOptions(temperature: 0.5, toolCallingMode: .disallowed)
                #expect(disallowed.toolCallingMode == .disallowed)

                let kinds: Set<GenerationOptions.ToolCallingMode.Kind> = [
                    GenerationOptions.ToolCallingMode.allowed.kind,
                    GenerationOptions.ToolCallingMode.required.kind,
                    GenerationOptions.ToolCallingMode.disallowed.kind,
                ]
                #expect(kinds == [.allowed, .required, .disallowed])

                var context = ContextOptions()
                #expect(context.includeSchemaInPrompt == nil)
                #expect(context.reasoningLevel == nil)
                context.includeSchemaInPrompt = true
                context.reasoningLevel = .custom("extended")
                #expect(context == ContextOptions(includeSchemaInPrompt: true, reasoningLevel: .custom("extended")))
                #expect(ContextOptions(reasoningLevel: .light) != ContextOptions(reasoningLevel: .moderate))
                #expect(ContextOptions(reasoningLevel: .deep) != ContextOptions())
            }
        #endif
    #endif
#endif
