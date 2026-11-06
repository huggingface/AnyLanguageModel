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

        // Foundation Models 27 evaluates a profile and runs its history transforms again
        // before each continuation, from the unprojected entries.
        @Test(arguments: [false, true])
        func continuationEvaluatesTheProfileAndItsTransformAgain(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let evaluations = Locked(0)
            let transforms = Locked(0)
            let transformInputs = Locked<[[String]]>([])
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    CountingProjectionInstructions(evaluations: evaluations)
                    WeatherTool()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { entries in
                    let count = transforms.withLock {
                        $0 += 1
                        return $0
                    }
                    transformInputs.withLock { $0.append(entries.compactMap(entryKind)) }
                    return entries.map { entry in
                        guard case .prompt(var prompt) = entry else { return entry }
                        prompt.segments = [.text(.init(content: "Rewrite \(count)"))]
                        return .prompt(prompt)
                    }
                }
            )
            let evaluationsAtCreation = evaluations.withLock { $0 }

            try await run(session, streaming: streaming)

            let sent = recording.sent.withLock { $0 }
            #expect(sent.map(\.instructions) == ["Revision 2", "Revision 3"])
            #expect(
                sent.map(\.entries) == [
                    ["prompt:Rewrite 1"],
                    ["prompt:Rewrite 2", "calls:weather-call", "output:weather-call"],
                ]
            )
            #expect(evaluations.withLock { $0 } == evaluationsAtCreation + 2)
            #expect(transforms.withLock { $0 } == 2)
            #expect(
                transformInputs.withLock { $0 } == [
                    ["prompt:Current"],
                    ["prompt:Current", "calls", "output"],
                ]
            )
            #expect(promptTexts(session.transcript) == ["Current"])
        }

        @Test(arguments: [false, true])
        func continuationOffersTheToolsOfItsOwnRevision(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    ToolPerRevisionInstructions(evaluations: evaluations)
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
            )

            try await run(session, streaming: streaming)

            #expect(
                recording.sent.withLock { $0 }.map(\.enabledTools) == [
                    ["getWeather"],
                    ["getWeather", "getTime"],
                ]
            )
        }

        @Test(arguments: [false, true])
        func parallelToolCallsEvaluateTheProfileOnceAndKeepTheModelsIDs(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true, parallelCalls: true)
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    CountingProjectionInstructions(evaluations: evaluations)
                    WeatherTool()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
            )
            let evaluationsAtCreation = evaluations.withLock { $0 }

            try await run(session, streaming: streaming)

            #expect(evaluations.withLock { $0 } == evaluationsAtCreation + 2)
            #expect(
                recording.sent.withLock { $0 }.last?.entries == [
                    "prompt:Current",
                    "calls:weather-call,weather-call-2",
                    "output:weather-call",
                    "output:weather-call-2",
                ]
            )
        }

        @Test(arguments: [false, true])
        func continuationCanStopRewritingThePrompt(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let transforms = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { entries in
                    let count = transforms.withLock {
                        $0 += 1
                        return $0
                    }
                    guard count == 1 else { return entries }
                    return entries.map { entry in
                        guard case .prompt = entry else { return entry }
                        return .prompt(.init(segments: [.text(.init(content: "Rewritten"))]))
                    }
                }
            )

            try await run(session, streaming: streaming)

            #expect(recording.sent.withLock { $0 }.map(\.entries.first) == ["prompt:Rewritten", "prompt:Current"])
        }

        @Test(arguments: [false, true])
        func transformCanEditAToolRound(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { entries in
                    entries.map { entry in
                        guard case .toolOutput(let output) = entry else { return entry }
                        return .toolOutput(
                            .init(
                                id: output.id,
                                toolName: output.toolName,
                                segments: [.text(.init(content: "Redacted"))]
                            )
                        )
                    }
                }
            )

            try await run(session, streaming: streaming)

            #expect(recording.sent.withLock { $0 }.last?.toolOutputs == ["Redacted"])
        }

        @Test(arguments: [false, true])
        func transformCanRemoveTheCurrentPrompt(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { entries in
                    entries.filter { entry in
                        if case .prompt = entry { return false }
                        return true
                    }
                }
            )

            try await run(session, streaming: streaming)

            #expect(
                recording.sent.withLock { $0 }.map(\.entries) == [
                    [],
                    ["calls:weather-call", "output:weather-call"],
                ]
            )
            #expect(promptTexts(session.transcript) == ["Current"])
        }

        @Test(arguments: [false, true])
        func transformCanMoveTheCurrentPrompt(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let earlier = AnyLanguageModel.Transcript.Entry.response(
                .init(assetIDs: [], segments: [.text(.init(content: "Earlier"))])
            )
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { entries in
                    func isPrompt(_ entry: AnyLanguageModel.Transcript.Entry) -> Bool {
                        if case .prompt = entry { return true }
                        return false
                    }
                    return entries.filter(isPrompt) + entries.filter { !isPrompt($0) }
                },
                history: [earlier]
            )

            try await run(session, streaming: streaming)

            #expect(
                recording.sent.withLock { $0 }.map(\.entries) == [
                    ["prompt:Current", "response"],
                    ["prompt:Current", "response", "calls:weather-call", "output:weather-call"],
                ]
            )
        }

        @Test(arguments: [false, true])
        func rewrittenPromptWithASchemaKeepsItsOptionsAndResponseFormatThroughAToolRound(
            streaming: Bool
        ) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true, responseText: #"{"name":"Rex"}"#)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform(replacingPrompts(with: [.text(.init(content: "Rewritten"))]))
            )
            let schema = try AnyLanguageModel.GenerationSchema(
                type: AnyLanguageModel.GeneratedContent.self,
                properties: [.init(name: "name", type: String.self)]
            )
            let options = AnyLanguageModel.GenerationOptions(temperature: 0.25)

            if streaming {
                for try await _ in session.streamResponse(to: "Current", schema: schema, options: options) {}
            } else {
                _ = try await session.respond(to: "Current", schema: schema, options: options)
            }

            let sent = recording.sent.withLock { $0 }
            #expect(sent.map(\.entries.first) == ["prompt:Rewritten", "prompt:Rewritten"])
            #expect(sent.map(\.promptTemperature) == [0.25, 0.25])
            #expect(sent.map(\.promptHasResponseFormat) == [true, true])
            #expect(sent.map(\.requestTemperature) == [0.25, 0.25])
            #expect(sent.map(\.requestHasSchema) == [true, true])
        }

        /// A direct history write can land while the model loads, after the request's context resolved.
        @Test(arguments: [false, true])
        func rewriteUsesTheHistoryOfTheRequestsContext(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel()
            let sessionBox = Locked<AnyLanguageModel.LanguageModelSession?>(nil)
            let direct = AnyLanguageModel.Transcript.Entry.response(
                .init(assetIDs: [], segments: [.text(.init(content: "Direct"))])
            )
            let model = AnyLanguageModel.FoundationLanguageModel<RecordingModel>(loading: {
                if let session = sessionBox.withLock({ $0 }) {
                    let live = session.transcript.filter { entry in
                        if case .instructions = entry { return false }
                        return true
                    }
                    session.properties.history = live + [direct]
                }
                return recording
            })
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    AnyLanguageModel.Instructions("Projected")
                }
                .model(model)
                .historyTransform(replacingPrompts(with: [.text(.init(content: "Rewritten"))]))
            )
            sessionBox.withLock { $0 = session }

            try await run(session, streaming: streaming)

            #expect(recording.requestTexts == [["Rewritten"]])
        }

        @Test(arguments: [false, true], [false, true])
        func continuationRewritingToARemoteImageFails(streaming: Bool, schema: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let url = try #require(URL(string: "https://example.com/image.png"))
            let transforms = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(AnyLanguageModel.FoundationLanguageModel(recording))
                .historyTransform { entries in
                    let count = transforms.withLock {
                        $0 += 1
                        return $0
                    }
                    guard count > 1 else { return entries }
                    return entries.map { entry in
                        guard case .prompt = entry else { return entry }
                        return .prompt(.init(segments: [.image(.init(url: url))]))
                    }
                }
            )
            let structure = try AnyLanguageModel.GenerationSchema(
                type: AnyLanguageModel.GeneratedContent.self,
                properties: [.init(name: "name", type: String.self)]
            )

            await #expect(throws: AnyLanguageModel.PromptBridgeError.self) {
                switch (streaming, schema) {
                case (false, false):
                    _ = try await session.respond(to: "Current")
                case (false, true):
                    _ = try await session.respond(to: "Current", schema: structure)
                case (true, false):
                    for try await _ in session.streamResponse(to: "Current") {}
                case (true, true):
                    for try await _ in session.streamResponse(to: "Current", schema: structure) {}
                }
            }
            #expect(recording.sent.withLock { $0 }.count == 1)
        }

        @Test(arguments: [false, true])
        func continuationRunsTheLifecycleActionsOfANewProfile(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let events = Locked<[String]>([])
            let recording = RecordingModel(callsTool: true, events: events)
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                profile: ProfilePerRevision(
                    evaluations: evaluations,
                    events: events,
                    model: AnyLanguageModel.FoundationLanguageModel(recording)
                )
            )

            try await run(session, streaming: streaming)

            #expect(
                events.withLock { $0 } == [
                    "activate:first", "request", "deactivate:first", "activate:second", "request",
                ]
            )
        }

        @Test(arguments: [false, true])
        func ordinaryDynamicInstructionsResolveAgainBeforeAContinuation(streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
            let recording = RecordingModel(callsTool: true)
            let evaluations = Locked(0)
            let session = AnyLanguageModel.LanguageModelSession(
                model: AnyLanguageModel.FoundationLanguageModel(recording),
                dynamicInstructions: ToolPerRevisionInstructions(evaluations: evaluations)
            )

            try await run(session, streaming: streaming)

            let sent = recording.sent.withLock { $0 }
            #expect(sent.count == 2)
            #expect(sent.first?.instructions != sent.last?.instructions)
            #expect(sent.last?.entries == ["prompt:Current", "calls:weather-call", "output:weather-call"])
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

    /// A short description of an entry that a history transform receives.
    private func entryKind(_ entry: AnyLanguageModel.Transcript.Entry) -> String? {
        switch entry {
        case .instructions: nil
        case .prompt(let prompt): "prompt:\(promptText(prompt.segments))"
        case .toolCalls: "calls"
        case .toolOutput: "output"
        case .response: "response"
        case .reasoning: nil
        }
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

    /// What Foundation Models sent to a model in one request.
    private struct SentRequest: Equatable, Sendable {
        var instructions: String?
        var entries: [String]
        var toolOutputs: [String]
        var enabledTools: [String]
        var promptTemperature: Double?
        var promptHasResponseFormat: Bool
        var requestTemperature: Double?
        var requestHasSchema: Bool
    }

    /// A model that records the prompts of each request that Foundation Models sends it.
    @available(macOS 27, iOS 27, visionOS 27, *)
    private struct RecordingModel: FoundationModels.LanguageModel {
        typealias Executor = RecordingExecutor
        let requests = Locked<[[[RecordedSegment]]]>([])
        let sent = Locked<[SentRequest]>([])
        var callsTool = false
        var parallelCalls = false
        var responseText = "OK"
        var events: Locked<[String]>?
        var capabilities: FoundationModels.LanguageModelCapabilities {
            .init([.vision, .toolCalling, .guidedGeneration])
        }
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
            model.sent.withLock { $0.append(SentRequest(request)) }
            model.events?.withLock { $0.append("request") }
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
                if model.parallelCalls {
                    await channel.send(
                        .toolCalls(
                            action: .toolCall(
                                id: "weather-call-2",
                                name: "getWeather",
                                action: .appendArguments(#"{"city":"Rome"}"#, tokenCount: 1)
                            )
                        )
                    )
                }
                return
            }
            await channel.send(.response(action: .appendText(model.responseText, tokenCount: 1)))
        }
    }

    @available(macOS 27, iOS 27, visionOS 27, *)
    extension SentRequest {
        init(_ request: FoundationModels.LanguageModelExecutorGenerationRequest) {
            func text(_ segments: [FoundationModels.Transcript.Segment]) -> String {
                segments.map { segment in
                    if case .text(let text) = segment { return text.content }
                    return "?"
                }.joined()
            }
            var instructions: String?
            var entries: [String] = []
            var toolOutputs: [String] = []
            var promptTemperature: Double?
            var promptHasResponseFormat = false
            for entry in request.transcript {
                switch entry {
                case .instructions(let value):
                    instructions = text(value.segments)
                case .prompt(let prompt):
                    entries.append("prompt:\(text(prompt.segments))")
                    promptTemperature = prompt.options.temperature
                    promptHasResponseFormat = prompt.responseFormat != nil
                case .toolCalls(let calls):
                    entries.append("calls:\(calls.map(\.id).joined(separator: ","))")
                case .toolOutput(let output):
                    entries.append("output:\(output.id)")
                    toolOutputs.append(text(output.segments))
                case .response:
                    entries.append("response")
                default:
                    entries.append("other")
                }
            }
            self.init(
                instructions: instructions,
                entries: entries,
                toolOutputs: toolOutputs,
                enabledTools: request.enabledToolDefinitions.map(\.name),
                promptTemperature: promptTemperature,
                promptHasResponseFormat: promptHasResponseFormat,
                requestTemperature: request.generationOptions.temperature,
                requestHasSchema: request.schema != nil
            )
        }
    }

    /// Offers one more tool from the second revision that a request sends.
    private struct ToolPerRevisionInstructions: AnyLanguageModel.DynamicInstructions {
        let evaluations: Locked<Int>

        var body: some AnyLanguageModel.DynamicInstructions {
            let count = evaluations.withLock {
                $0 += 1
                return $0
            }
            AnyLanguageModel.Instructions("Revision \(count)")
            WeatherTool()
            if count > 2 {
                ClockTool()
            }
        }
    }

    /// Selects another profile after the first revision that a request sends.
    private struct ProfilePerRevision: AnyLanguageModel.LanguageModelSession.DynamicProfile {
        let evaluations: Locked<Int>
        let events: Locked<[String]>
        let model: any AnyLanguageModel.LanguageModel

        var body: some AnyLanguageModel.LanguageModelSession.DynamicProfile {
            let count = evaluations.withLock {
                $0 += 1
                return $0
            }
            if count > 2 {
                AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(model)
                .onActivate { [events] in events.withLock { $0.append("activate:second") } }
                .onDeactivate { [events] in events.withLock { $0.append("deactivate:second") } }
            } else {
                AnyLanguageModel.LanguageModelSession.Profile {
                    WeatherInstructions()
                }
                .model(model)
                .onActivate { [events] in events.withLock { $0.append("activate:first") } }
                .onDeactivate { [events] in events.withLock { $0.append("deactivate:first") } }
            }
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
