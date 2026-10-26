import Foundation
import Testing

#if canImport(ImageIO)
    import CoreGraphics
    import ImageIO
#endif

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

    /// Takes no empty value: its sky has to be one of its cases.
    @Generable
    private struct Forecast {
        var sky: Sky

        @Generable
        enum Sky {
            case sunny, cloudy
        }
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
    private enum CounterStep {
        case add(amount: Int)
        case reset
        case report(String)
    }

    @Generable
    private struct CounterPlan {
        @Guide(description: "The steps, in order.")
        var steps: [CounterStep]
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

    /// A stop before any content, for a content type that takes no empty value, throws
    /// `typeMismatch`, as other models do, whether streaming or not. No model is involved.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test(arguments: [false, true])
    func stopWithoutEmptyContentThrowsTypeMismatch(streaming: Bool) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
        session.toolExecutionDelegate = StoppingDelegate()
        let makeSession: @Sendable () async throws -> FoundationModels.LanguageModelSession = {
            _ = try await callTool(weatherTool, with: GeneratedContent(properties: ["city": "Paris"]), in: session)
            throw CancellationError()  // The call must stop before reaching here.
        }
        let error = await #expect(throws: GeneratedContentError.self) {
            if streaming {
                let stream = fmStreamResponse(
                    makeSession: makeSession,
                    fmPrompt: .init(""),
                    fmOptions: .init(),
                    type: Forecast.self,
                    schema: Forecast.generationSchema,
                    includeSchemaInPrompt: false
                )
                for try await _ in stream {}
            } else {
                _ = try await fmRespond(
                    makeSession: makeSession,
                    fmPrompt: .init(""),
                    fmOptions: .init(),
                    type: Forecast.self,
                    schema: Forecast.generationSchema,
                    includeSchemaInPrompt: false
                )
            }
        }
        guard case .typeMismatch? = error else {
            Issue.record("Expected typeMismatch, got \(String(describing: error))")
            return
        }
    }

    /// Parallel calls keep the order they started in, not the order they finish in.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test func callsKeepTheOrderTheyStartedIn() async {
        let record = ToolCallRecord()
        let arguments = GeneratedContent(properties: [:])
        let first = Transcript.ToolCall(id: "first", toolName: "getWeather", arguments: arguments)
        let second = Transcript.ToolCall(id: "second", toolName: "getWeather", arguments: arguments)
        await record.start(first)
        await record.start(second)
        for call in [second, first] {
            await record.ran(
                call,
                output: Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: [.text(.init(content: "Sunny"))]
                )
            )
        }
        let order = await record.entries.compactMap { entry -> String? in
            if case .toolOutput(let output) = entry { return output.id }
            return nil
        }
        #expect(order == ["first", "second"])
    }

    /// An image the delegate provides that Foundation Models can't take fails the call,
    /// rather than reaching the model as nothing while the transcript shows it given.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test(arguments: [
        Transcript.ImageSegment(url: URL(string: "https://example.com/forecast.png")!),
        Transcript.ImageSegment(data: Data("not an image".utf8), mimeType: "image/png"),
    ])
    func providedImageFoundationModelsCannotTakeFailsTheCall(image: Transcript.ImageSegment) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
        session.toolExecutionDelegate = ScriptedDelegate([.provideOutput([.image(image)])])
        let record = ToolCallRecord()
        await ToolCallRecord.$current.withValue(record) {
            await #expect(throws: PromptBridgeError.self) {
                _ = try await callTool(weatherTool, with: GeneratedContent(properties: ["city": "Paris"]), in: session)
            }
        }
        #expect(await record.entries.isEmpty)
    }

    #if compiler(>=6.4)
        /// An image the delegate provides goes to Foundation Models as an attachment.
        @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
        @Test func providedImageIsAttached() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])
            let image = Transcript.ImageSegment(data: try onePixelPNG(), mimeType: "image/png")
            session.toolExecutionDelegate = ScriptedDelegate([.provideOutput([.image(image)])])
            let record = ToolCallRecord()
            try await ToolCallRecord.$current.withValue(record) {
                _ = try await callTool(weatherTool, with: GeneratedContent(properties: ["city": "Paris"]), in: session)
            }
            let outputs = await record.entries.compactMap { entry -> Transcript.ToolOutput? in
                if case .toolOutput(let output) = entry { return output }
                return nil
            }
            #expect(outputs.map(\.segments) == [[.image(image)]])
        }

        private func onePixelPNG() throws -> Data {
            let context = try #require(
                CGContext(
                    data: nil,
                    width: 1,
                    height: 1,
                    bitsPerComponent: 8,
                    bytesPerRow: 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                )
            )
            let image = try #require(context.makeImage())
            let data = NSMutableData()
            let destination = try #require(CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            try #require(CGImageDestinationFinalize(destination))
            return data as Data
        }
    #endif

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
            attachResponse(response)
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
            attachResponse(response)
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
            attachResponse(response)
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
                attachResponse(first, "first")
                #expect(first.content.localizedCaseInsensitiveContains("apple"))

                state.useWeather = true
                let second = try await session.respond(to: "How's the weather in San Francisco?", options: options)
                attachResponse(second, "second")
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
            attachResponse(response)

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
            attachResponse(response)

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
            attachResponse(response)

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
            attachResponse(firstResponse, "firstResponse")
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
            attachResponse(secondResponse, "secondResponse")
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
            attachResponse(response)

            #expect(!response.content.message.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithMultipleFields() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Create a fictional person who is a software engineer",
                generating: Person.self
            )
            attachResponse(response)

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
            attachResponse(response)

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
            attachResponse(response)

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
            attachResponse(response)

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
            attachResponse(response)

            #expect(["positive", "negative", "neutral"].contains(response.content.sentiment.lowercased()))
            #expect(response.content.confidence >= 0.0 && response.content.confidence <= 1.0)
        }

        /// An enum with associated values reaches Foundation Models as one object per case.
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithEnumPayloads() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Plan a counter: add 2, then add 3, then report the total.",
                generating: CounterPlan.self
            )
            attachResponse(response)

            #expect(!response.content.steps.isEmpty)
        }

        /// An enum with associated values can be the whole response,
        /// with a union of its cases at the root of the schema.
        @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithEnumPayloadRoot() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Add 2 to the counter.",
                generating: CounterStep.self
            )
            attachResponse(response)

            let type = try response.rawContent.value(String.self, forProperty: "type")
            #expect(["add", "reset", "report"].contains(type))
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
            attachResponse(response)

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

    /// What a tool asks of a model is its own, when Foundation Models calls it too:
    /// a request the tool makes to another model isn't counted for the response that called it,
    /// whether the session has a tool execution delegate or not. No system model is involved.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test(arguments: [false, true])
    func whatASystemModelToolUsesIsntCountedForTheResponseThatCalledIt(withDelegate: Bool) async throws {
        let tool = AskingOllama()
        let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [tool])
        let delegate = ScriptedDelegate([.execute])
        if withDelegate { session.toolExecutionDelegate = delegate }
        let channel = RequestUsageChannel { _ in }
        try await RequestUsageChannel.$current.withValue(channel) {
            _ = try await callTool(tool, with: GeneratedContent(properties: ["term": "tea"]), in: session)
        }
        #expect(tool.transport.handled == 1)
        #expect(channel.reportedUsage == .zero)
    }

    /// A tool that asks an Ollama model itself, which reports what the request used.
    private struct AskingOllama: Tool {
        let name = "lookup"
        let description = "Looks something up."
        let transport = FakeTransport(replies: [
            .init(chunks: [
                #"{"model":"m","created_at":"2026-10-22T10:00:00.000Z","message":{"role":"assistant","content":"Found"},"done":true,"prompt_eval_count":7,"eval_count":3}"#
            ])
        ])

        @Generable struct Arguments { var term: String }

        func call(arguments: Arguments) async throws -> String {
            let model = OllamaLanguageModel(
                baseURL: URL(string: "http://localhost:11434")!,
                model: "m",
                transport: transport
            )
            let response = try await model.respond(
                within: LanguageModelSession(model: model),
                to: Prompt(arguments.term),
                generating: String.self,
                includeSchemaInPrompt: false,
                options: GenerationOptions()
            )
            return response.content
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
