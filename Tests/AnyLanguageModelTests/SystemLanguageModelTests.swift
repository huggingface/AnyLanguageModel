import Foundation
import Testing
@testable import AnyLanguageModel

#if canImport(FoundationModels) && !os(watchOS)
    import class FoundationModels.LanguageModelSession

    private typealias LanguageModelSession = AnyLanguageModel.LanguageModelSession

    private let isSystemLanguageModelAvailable = {
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return SystemLanguageModel.default.isAvailable
        } else {
            return false
        }
    }()

    // MARK: - Test Types for Guided Generation

    @Generable
    private struct Greeting {
        @Guide(description: "A greeting message")
        var message: String
    }

    @Generable
    private struct Person {
        @Guide(description: "The person's full name")
        var name: String

        @Guide(description: "The person's age in years", .range(0 ... 150))
        var age: Int

        @Guide(description: "The person's occupation")
        var occupation: String
    }

    @Generable
    private struct MathResult {
        @Guide(description: "The mathematical expression that was evaluated")
        var expression: String

        @Guide(description: "The numeric result of the calculation")
        var result: Int

        @Guide(description: "Step-by-step explanation of how the result was calculated")
        var explanation: String
    }

    @Generable
    private struct ColorInfo {
        @Guide(description: "The name of the color")
        var name: String

        @Guide(description: "The hex code for the color, e.g. #FF0000")
        var hexCode: String

        @Guide(description: "RGB values for the color")
        var rgb: RGBValues
    }

    @Generable
    private struct RGBValues {
        @Guide(description: "Red component (0-255)", .range(0 ... 255))
        var red: Int

        @Guide(description: "Green component (0-255)", .range(0 ... 255))
        var green: Int

        @Guide(description: "Blue component (0-255)", .range(0 ... 255))
        var blue: Int
    }

    @Generable
    private struct BookRecommendations {
        @Guide(description: "List of recommended book titles")
        var titles: [String]
    }

    @Generable
    private struct SentimentAnalysis {
        @Guide(description: "The sentiment classification", .anyOf(["positive", "negative", "neutral"]))
        var sentiment: String

        @Guide(description: "Confidence score between 0 and 1")
        var confidence: Double
    }

    // MARK: - Test Suite

    /// A tool called for Foundation Models asks the delegate first, and a stop ends the call
    /// without running the tool. No model is involved, so this runs wherever Foundation Models builds.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test func stoppedToolCallDoesNotRunTheTool() async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
        let delegate = StoppingDelegate()
        session.toolExecutionDelegate = delegate
        await #expect(throws: ToolCallStopped.self) {
            _ = try await callTool(weatherTool, with: try GeneratedContent(json: #"{"city":"Cupertino"}"#), in: session)
        }
        #expect(await weatherTool.calls.isEmpty)
        let generated = await delegate.generated
        #expect(generated.map(\.toolName) == [weatherTool.name])
        #expect(generated.first?.arguments.jsonString.contains("Cupertino") == true)
    }

    /// The calls that ran in a request before the delegate stopped at one stay in its transcript,
    /// with their outputs and the IDs the delegate saw. No model is involved.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test func callsThatRanBeforeAStopAreRecorded() async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
        let delegate = ScriptedDelegate([.provideOutput([.text(.init(content: "Sunny"))]), .stop])
        session.toolExecutionDelegate = delegate
        let record = ToolCallRecord()
        await ToolCallRecord.$current.withValue(record) {
            _ = try? await callTool(
                weatherTool,
                with: try! GeneratedContent(json: #"{"city":"Cupertino"}"#),
                in: session
            )
            _ = try? await callTool(weatherTool, with: try! GeneratedContent(json: #"{"city":"Paris"}"#), in: session)
        }
        let generated = await delegate.generated
        let entries = await record.entries
        #expect(entries.count == 2)
        guard case .toolCalls(let calls) = entries.first, case .toolOutput(let output) = entries.last else {
            Issue.record("Expected the first call and its output, got \(entries)")
            return
        }
        #expect(calls.map(\.id) == [generated[0].id])
        #expect(output.id == generated[0].id)
        #expect(output.segments.description.contains("Sunny"))
        #expect(await weatherTool.calls.isEmpty)
    }

    /// Exercise the response wrappers without requiring model inference.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test(arguments: [false, true])
    func stoppedResponsePreservesCompletedToolCalls(streaming: Bool) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = AnyLanguageModel.LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
        let delegate = ScriptedDelegate([.execute, .provideOutput([.text(.init(content: "Sunny"))]), .stop])
        session.toolExecutionDelegate = delegate
        let makeSession: @Sendable () async throws -> FoundationModels.LanguageModelSession = {
            for city in ["Cupertino", "Paris", "Berlin"] {
                _ = try await callTool(weatherTool, with: GeneratedContent(properties: ["city": city]), in: session)
            }
            throw CancellationError()  // The third call must stop before reaching here.
        }
        let entries: [AnyLanguageModel.Transcript.Entry]
        if streaming {
            let stream = fmStreamResponse(
                makeSession: makeSession,
                fmPrompt: .init(""),
                fmOptions: .init(),
                type: String.self,
                schema: String.generationSchema,
                includeSchemaInPrompt: false
            )
            var last: AnyLanguageModel.LanguageModelSession.ResponseStream<String>.Snapshot?
            for try await snapshot in stream { last = snapshot }
            entries = Array(try #require(last).transcriptEntries)
        } else {
            let response = try await fmRespond(
                makeSession: makeSession,
                fmPrompt: .init(""),
                fmOptions: .init(),
                type: String.self,
                schema: String.generationSchema,
                includeSchemaInPrompt: false
            )
            entries = Array(response.transcriptEntries)
        }
        let generated = await delegate.generated
        try #require(generated.count == 3)
        try #require(entries.count == 5)
        for index in 0 ..< 2 {
            guard case .toolCalls(let calls) = entries[index * 2],
                case .toolOutput(let output) = entries[index * 2 + 1]
            else {
                Issue.record("Expected a completed call followed by its output")
                return
            }
            #expect(calls.map(\.id) == [generated[index].id])
            #expect(output.id == generated[index].id)
        }
        guard case .toolCalls(let stopped) = entries[4] else {
            Issue.record("Expected the stopped call last")
            return
        }
        #expect(stopped.map(\.id) == [generated[2].id])
        #expect(await weatherTool.calls.count == 1)
    }

    /// A stop is found inside an error that wraps it, as tool errors are wrapped.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test func stopIsFoundInsideToolCallErrors() {
        let call = Transcript.ToolCall(id: "call", toolName: "getWeather", arguments: GeneratedContent(properties: [:]))
        let wrapped = LanguageModelSession.ToolCallError(
            tool: WeatherTool(),
            underlyingError: ToolCallStopped(call: call)
        )
        #expect(stoppedToolCall(in: wrapped)?.id == "call")
        #expect(stoppedToolCall(in: CancellationError()) == nil)
    }

    @Test("GenerationSchema merges duplicate defs for the same type")
    func generationSchemaMergesDuplicateDefsForSameType() {
        let schema = ContainerWithDuplicateNestedType.generationSchema

        let nestedTypeName = String(reflecting: ReusedNestedStruct.self)
        #expect(schema.defs[nestedTypeName] != nil)
    }

    private final class SwitchState: @unchecked Sendable {
        var useWeather = false
    }

    private struct SwitchedInstructions: DynamicInstructions {
        let state: SwitchState

        var body: some DynamicInstructions {
            if state.useWeather {
                Instructions("Answer weather questions with the getWeather tool.")
                WeatherTool()
            } else {
                Instructions("Reply with exactly one word: apple.")
            }
        }
    }

    private struct BriefInstructions: DynamicInstructions {
        var body: some DynamicInstructions {
            Instructions("Be brief.")
        }
    }

    /// Before OS 27, and on tvOS, the system model rejects dynamic instructions
    /// before it uses the model, so this runs even where the model is unavailable.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test func dynamicInstructionsAreUnavailableWithoutNativeSupport() async throws {
        var hasNativeSupport = false
        #if compiler(>=6.4) && !os(tvOS)
            if #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) {
                hasNativeSupport = true
            }
        #endif
        guard !hasNativeSupport else { return }

        let session = LanguageModelSession(model: SystemLanguageModel(), dynamicInstructions: BriefInstructions())
        await #expect(throws: SystemLanguageModel.Error.dynamicInstructionsUnavailable) {
            try await session.respond(to: "Hello")
        }
        await #expect(throws: SystemLanguageModel.Error.dynamicInstructionsUnavailable) {
            for try await _ in session.streamResponse(to: "Hello") {}
        }
    }

    @Suite(
        "SystemLanguageModel",
        .enabled(if: isSystemLanguageModelAvailable)
    )
    struct SystemLanguageModelTests {
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func supportedLanguagesIncludeEnglish() {
            let model = SystemLanguageModel()
            #expect(model.supportedLanguages.contains { $0.languageCode == .english })
            #expect(model.supportsLocale(Locale(identifier: "en_US")))
        }

        #if compiler(>=6.3) && !os(tvOS)
            @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
            @Test func tokenCountsArePositive() async throws {
                guard #available(macOS 26.4, iOS 26.4, visionOS 26.4, *) else { return }
                let model = SystemLanguageModel()

                let short = try await model.tokenCount(for: "Hello")
                let long = try await model.tokenCount(for: Prompt("How's the weather in Cupertino today?"))
                #expect(short > 0)
                #expect(long > short)

                #expect(try await model.tokenCount(for: Instructions("Answer in one sentence.")) > 0)

                #expect(try await model.tokenCount(for: [WeatherTool()]) > 0)
                #expect(try await model.tokenCount(for: Person.generationSchema) > 0)

                let entries: [Transcript.Entry] = [
                    .prompt(Transcript.Prompt(segments: [.text(.init(content: "Hello"))])),
                    .response(Transcript.Response(assetIDs: [], segments: [.text(.init(content: "Hi there!"))])),
                ]
                #expect(try await model.tokenCount(for: entries) > 0)
            }
        #endif

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func basicResponse() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say 'Hello'")
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func withInstructions() async throws {
            let model = SystemLanguageModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func withTemperature() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let options = AnyLanguageModel.GenerationOptions(temperature: 0.5)
            let response = try await session.respond(
                to: "Generate a number",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func streamingString() async throws {
            guard isSystemLanguageModelAvailable else { return }
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 20 in Italian")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func streamingGeneratedContent() async throws {
            guard isSystemLanguageModelAvailable else { return }
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(
                to: Prompt("Provide a JSON object with a field 'text'"),
                schema: GeneratedContent.generationSchema
            )

            var snapshots: [LanguageModelSession.ResponseStream<GeneratedContent>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        #if compiler(>=6.4) && !os(tvOS)
            @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
            @Test func dynamicInstructionsChangeBetweenRequests() async throws {
                guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else { return }
                let state = SwitchState()
                let session = LanguageModelSession(
                    model: SystemLanguageModel(),
                    dynamicInstructions: SwitchedInstructions(state: state)
                )
                let options = GenerationOptions(sampling: .greedy)

                let first = try await session.respond(to: "What should you reply?", options: options)
                #expect(first.content.localizedCaseInsensitiveContains("apple"))

                state.useWeather = true
                let second = try await session.respond(to: "How's the weather in San Francisco?", options: options)
                #expect(second.content.contains("72°F"))

                #expect(
                    !session.transcript.contains {
                        if case .instructions = $0 { true } else { false }
                    }
                )
            }
        #endif

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func withTools() async throws {
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            // The calls Foundation Models ran are in the response's transcript, with their outputs.
            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.toolName == weatherTool.name)
                foundToolOutput = true
            }
            #expect(foundToolOutput)

            let content = response.content
            #expect(content.contains("San Francisco"))
            #expect(content.contains("72°F"))
        }

        /// A delegate that stops at tool calls gets them back instead of Foundation Models running them,
        /// as with other models.
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func delegateStopsToolCalls() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
            let delegate = StoppingDelegate()
            session.toolExecutionDelegate = delegate

            let response = try await session.respond(to: "How's the weather in San Francisco? Use the tool.")

            #expect(await weatherTool.calls.isEmpty)
            let generated = await delegate.generated
            #expect(generated.map(\.toolName) == [weatherTool.name])
            #expect(
                response.transcriptEntries.contains { entry in
                    if case .toolCalls(let calls) = entry { return calls.map(\.id) == generated.map(\.id) }
                    return false
                }
            )
        }

        /// Output the delegate provides is what Foundation Models goes on with, in the same request,
        /// and the call and its output are in the response's transcript, with the delegate's IDs.
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func delegateProvidesToolOutput() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
            let delegate = ScriptedDelegate([])
            session.toolExecutionDelegate = delegate

            let response = try await session.respond(to: "How's the weather in San Francisco? Use the tool.")

            #expect(await weatherTool.calls.isEmpty)
            let generated = await delegate.generated
            #expect(!generated.isEmpty)
            let outputs = response.transcriptEntries.compactMap { entry -> Transcript.ToolOutput? in
                if case .toolOutput(let output) = entry { return output }
                return nil
            }
            #expect(outputs.map(\.id) == generated.map(\.id))
            #expect(response.content.contains("20°F"))
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func delegateProvidesToolOutputWhileStreaming() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
            let delegate = ScriptedDelegate([])
            session.toolExecutionDelegate = delegate

            var last: LanguageModelSession.ResponseStream<String>.Snapshot?
            for try await snapshot in session.streamResponse(to: "How's the weather in San Francisco? Use the tool.") {
                last = snapshot
            }

            #expect(await weatherTool.calls.isEmpty)
            let generated = await delegate.generated
            let outputs = try #require(last).transcriptEntries.compactMap { entry -> Transcript.ToolOutput? in
                if case .toolOutput(let output) = entry { return output }
                return nil
            }
            #expect(!generated.isEmpty)
            #expect(outputs.map(\.id) == generated.map(\.id))
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func delegateStopsToolCallsWhileStreaming() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
            let delegate = StoppingDelegate()
            session.toolExecutionDelegate = delegate

            var last: LanguageModelSession.ResponseStream<String>.Snapshot?
            for try await snapshot in session.streamResponse(to: "How's the weather in San Francisco? Use the tool.") {
                last = snapshot
            }

            #expect(await weatherTool.calls.isEmpty)
            #expect(await delegate.generated.map(\.toolName) == [weatherTool.name])
            let entries = try #require(last).transcriptEntries
            #expect(
                entries.contains { entry in
                    if case .toolCalls = entry { return true }
                    return false
                }
            )
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func conversationContext() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let numbers = (0 ..< 3).map { _ in Int.random(in: 1 ... 100) }
            let payload = numbers.map(String.init).joined(separator: ", ")
            let promptText: () -> String = {
                session.transcript.compactMap { entry -> String? in
                    guard case let .prompt(prompt) = entry else {
                        return nil
                    }
                    return prompt.segments.compactMap { segment -> String? in
                        guard case let .text(text) = segment else {
                            return nil
                        }
                        return text.content
                    }
                    .joined(separator: " ")
                }
                .joined(separator: " ")
            }
            let isGuardrailViolation: (any Error) -> Bool = { error in
                guard let generationError = error as? LanguageModelSession.GenerationError else {
                    return false
                }
                if case .guardrailViolation = generationError {
                    return true
                }
                return false
            }

            let firstResponse: LanguageModelSession.Response<String>
            do {
                firstResponse = try await session.respond(
                    to: "Remember these numbers: \(payload). Reply with just the numbers."
                )
            } catch {
                if isGuardrailViolation(error) {
                    #expect(promptText().contains(payload))
                    return
                }
                throw error
            }
            #expect(!firstResponse.content.isEmpty)

            let secondResponse: LanguageModelSession.Response<String>
            do {
                secondResponse = try await session.respond(
                    to: "What numbers did I ask you to remember? Reply with just the numbers."
                )
            } catch {
                if isGuardrailViolation(error) {
                    #expect(promptText().contains(payload))
                    return
                }
                throw error
            }
            let repliedNumbers = secondResponse.content
                .split { !$0.isNumber }
                .compactMap { Int($0) }
            if Set(repliedNumbers) != Set(numbers) {
                // Guardrails can refuse to repeat exact values
                // Verify the prompt was stored instead.
                #expect(session.transcript.count >= 4)
                #expect(promptText().contains(payload))
            }
        }

        // MARK: - Guided Generation Tests

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationSimpleStruct() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Generate a friendly greeting",
                generating: Greeting.self
            )

            #expect(!response.content.message.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithMultipleFields() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Create a fictional person who is a software engineer",
                generating: Person.self
            )

            #expect(!response.content.name.isEmpty)
            #expect(response.content.age >= 0 && response.content.age <= 150)
            #expect(!response.content.occupation.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationMathCalculation() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Calculate 15 + 27",
                generating: MathResult.self
            )

            #expect(!response.content.expression.isEmpty)
            #expect(!response.content.explanation.isEmpty)
            let combined = response.content.expression + " " + response.content.explanation
            #expect(combined.contains("15") || combined.contains("27") || combined.contains("42"))
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationNestedStruct() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Describe the color red",
                generating: ColorInfo.self
            )

            #expect(!response.content.name.isEmpty)
            #expect(!response.content.hexCode.isEmpty)
            #expect(response.content.rgb.red >= 0 && response.content.rgb.red <= 255)
            #expect(response.content.rgb.green >= 0 && response.content.rgb.green <= 255)
            #expect(response.content.rgb.blue >= 0 && response.content.rgb.blue <= 255)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithArray() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Recommend 3 classic science fiction books",
                generating: BookRecommendations.self
            )

            if response.content.titles.isEmpty {
                #expect(response.rawContent.jsonString.contains("titles"))
            } else {
                #expect(response.content.titles.count >= 1)
            }
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithEnumConstraint() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Analyze the sentiment of: 'I love this product!'",
                generating: SentimentAnalysis.self
            )

            #expect(["positive", "negative", "neutral"].contains(response.content.sentiment.lowercased()))
            #expect(response.content.confidence >= 0.0 && response.content.confidence <= 1.0)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithInstructions() async throws {
            let session = LanguageModelSession(
                model: SystemLanguageModel.default,
                instructions: "You are a creative writing assistant. Be imaginative and detailed."
            )

            let response = try await session.respond(
                to: "Create an interesting fictional character",
                generating: Person.self
            )

            #expect(!response.content.name.isEmpty)
            #expect(response.content.age >= 0)
            #expect(!response.content.occupation.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationStreaming() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let stream = session.streamResponse(
                to: "Generate a greeting",
                generating: Greeting.self
            )

            var snapshots: [LanguageModelSession.ResponseStream<Greeting>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            if let lastSnapshot = snapshots.last {
                #expect(!lastSnapshot.rawContent.jsonString.isEmpty)
            }
        }
    }

    /// Decides each tool call in turn from `decisions`, then provides "Snowing" for the rest,
    /// recording each call.
    private actor ScriptedDelegate: ToolExecutionDelegate {
        private var decisions: [ToolExecutionDecision]
        private(set) var generated: [Transcript.ToolCall] = []

        init(_ decisions: [ToolExecutionDecision]) {
            self.decisions = decisions
        }

        func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {
            generated.append(contentsOf: toolCalls)
        }

        func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
            -> ToolExecutionDecision
        {
            decisions.isEmpty ? .provideOutput([.text(.init(content: "Snowing, 20°F"))]) : decisions.removeFirst()
        }
    }

    /// Stops at every tool call, recording it.
    private actor StoppingDelegate: ToolExecutionDelegate {
        private(set) var generated: [Transcript.ToolCall] = []

        func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {
            generated.append(contentsOf: toolCalls)
        }

        func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
            -> ToolExecutionDecision
        {
            .stop
        }
    }
#endif
