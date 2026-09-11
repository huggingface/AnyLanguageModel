import Foundation
import Testing

@testable import AnyLanguageModel

#if MLX
    private let shouldRunMLXTests = {
        // Enable when explicitly requested via environment variable
        if ProcessInfo.processInfo.environment["ENABLE_MLX_TESTS"] != nil {
            return true
        }

        // Skip in CI environments
        if ProcessInfo.processInfo.environment["CI"] != nil {
            return false
        }

        // Skip unless Hugging Face API token is provided
        if ProcessInfo.processInfo.environment["HF_TOKEN"] == nil {
            return false
        }

        // Enable when running with Xcode/xcodebuild
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            return true
        }

        // Skip by default when running with swift test
        return false
    }()

    @Suite("MLXLanguageModel", .enabled(if: shouldRunMLXTests), .serialized)
    struct MLXLanguageModelTests {
        // Qwen3-0.6B is a small model that supports tool calling
        let model = MLXLanguageModel(
            modelId: "mlx-community/Qwen3-0.6B-4bit",
            directory: ProcessInfo.processInfo.environment["MLX_MODEL_DIRECTORY"].map { URL(fileURLWithPath: $0) }
        )
        let visionModel = MLXLanguageModel(modelId: "mlx-community/Qwen2-VL-2B-Instruct-4bit")

        @Test func availabilityBecomesAvailableAfterSuccessfulLoad() async throws {
            await model.removeFromCache()

            #expect(model.availability == .unavailable(.notLoaded))
            #expect(model.isAvailable == false)

            let session = LanguageModelSession(model: model)
            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)

            #expect(model.availability == .available)
            #expect(model.isAvailable == true)
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func streamingResponse() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func tokenUsageAndCacheReuse() async throws {
            let options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 8)
            let session = LanguageModelSession(model: model)
            let first = try await session.respond(to: "Say hello.", options: options)
            #expect(first.usage.input.totalTokenCount > 0)
            #expect(first.usage.input.cachedTokenCount == 0)
            #expect(first.usage.output.totalTokenCount > 0)
            #expect(first.usage.output.totalTokenCount <= 8)

            let second = try await session.respond(to: "Say goodbye.", options: options)
            #expect(second.usage.input.cachedTokenCount == first.usage.input.totalTokenCount)
            #expect(second.usage.input.totalTokenCount > second.usage.input.cachedTokenCount)
            #expect(session.usage.totalTokenCount == first.usage.totalTokenCount + second.usage.totalTokenCount)

            let fresh = LanguageModelSession(model: model)
            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in fresh.streamResponse(to: "Say hello.", options: options) {
                snapshots.append(snapshot)
            }
            let final = try #require(snapshots.last)
            #expect(final.usage == first.usage)
            #expect(final.content == first.content)
            #expect(fresh.usage == final.usage)
            #expect(snapshots.count >= 2)
            #expect(snapshots.dropLast().last?.rawContent == final.rawContent)
            #expect(snapshots.dropLast().last?.usage == .zero)
        }

        @Test func tokenUsageStructuredResponseStreamParity() async throws {
            let options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 16)
            let response = try await LanguageModelSession(model: model).respond(
                to: "Return true.",
                generating: Bool.self,
                options: options
            )
            let session = LanguageModelSession(model: model)
            let streamed = try await session.streamResponse(
                to: "Return true.",
                generating: Bool.self,
                options: options
            ).collect()
            #expect(response.usage.input.totalTokenCount > 0)
            #expect(response.usage.output.totalTokenCount > 0)
            #expect(response.usage.input.cachedTokenCount == 0)
            #expect(streamed.content == response.content)
            #expect(streamed.usage == response.usage)
            #expect(session.usage == streamed.usage)

            let withoutSchema = try await LanguageModelSession(model: model).respond(
                to: "Return true.",
                generating: Bool.self,
                includeSchemaInPrompt: false,
                options: options
            )
            #expect(response.usage.input.totalTokenCount > withoutSchema.usage.input.totalTokenCount)
        }

        @Test func tokenUsageSurvivesToolStopAndFailure() async throws {
            var options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)
            var custom = MLXLanguageModel.CustomGenerationOptions.default
            custom.additionalContext = ["enable_thinking": .bool(false)]
            options[custom: MLXLanguageModel.self] = custom
            let prompt = "Use getWeather to get the weather in San Francisco."
            let stopped = LanguageModelSession(model: model, tools: [WeatherTool()])
            stopped.toolExecutionDelegate = UsageStopDelegate()
            let response = try await stopped.respond(to: prompt, options: options)
            #expect(response.transcriptEntries.contains { if case .toolCalls = $0 { true } else { false } })
            #expect(response.usage.input.totalTokenCount > 0)
            #expect(response.usage.output.totalTokenCount > 0)

            let streamed = LanguageModelSession(model: model, tools: [WeatherTool()])
            streamed.toolExecutionDelegate = UsageStopDelegate()
            let collected = try await streamed.streamResponse(to: prompt, options: options).collect()
            #expect(collected.usage == response.usage)
            #expect(streamed.usage == response.usage)

            let completed = LanguageModelSession(model: model, tools: [WeatherTool()])
            var reportedRounds: [LanguageModelSession.Usage] = []
            for try await snapshot in completed.streamResponse(to: prompt, options: options) {
                if snapshot.usage != .zero, snapshot.usage != reportedRounds.last {
                    reportedRounds.append(snapshot.usage)
                }
            }
            #expect(reportedRounds.count == 2)
            #expect(reportedRounds.first == response.usage)
            let total = try #require(reportedRounds.last)
            #expect(total.input.totalTokenCount > response.usage.input.totalTokenCount)
            #expect(total.output.totalTokenCount > response.usage.output.totalTokenCount)
            #expect(completed.usage == total)
            let fullResponse = try await LanguageModelSession(model: model, tools: [WeatherTool()]).respond(
                to: prompt,
                options: options
            )
            #expect(fullResponse.usage == total)

            let failing = LanguageModelSession(model: model, tools: [UsageFailingWeatherTool()])
            await #expect(throws: LanguageModelSession.ToolCallError.self) {
                try await failing.streamResponse(to: prompt, options: options).collect()
            }
            #expect(failing.usage == response.usage)
        }

        private struct UsageStopDelegate: ToolExecutionDelegate {
            func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
                -> ToolExecutionDecision
            {
                .stop
            }
        }

        private struct UsageFailingWeatherTool: Tool {
            let name = WeatherTool().name
            let description = WeatherTool().description
            struct Failure: Error {}
            func call(arguments: WeatherTool.Arguments) async throws -> String { throw Failure() }
        }

        @Test func multiTurnSameSession() async throws {
            let session = LanguageModelSession(model: model)
            let first = try await session.respond(to: "Say hello in one sentence.")
            #expect(!first.content.isEmpty)

            let second = try await session.respond(to: "Now answer with one more short sentence.")
            #expect(!second.content.isEmpty)
        }

        @Test func rejectsConcurrentRequestsForSameSession() async throws {
            let session = LanguageModelSession(model: model)
            let stream = session.streamResponse(
                to: "Count from 1 to 400 with one number per line.",
                options: .init(maximumResponseTokens: 256)
            )

            do {
                _ = try await session.respond(to: "This concurrent request should fail.")
                Issue.record("Expected concurrent request to throw.")
            } catch let error as LanguageModelSession.GenerationError {
                switch error {
                case .concurrentRequests:
                    break
                default:
                    Issue.record("Expected .concurrentRequests, got \(error)")
                }
            } catch {
                Issue.record("Expected GenerationError.concurrentRequests, got \(error)")
            }

            for try await _ in stream {
                break
            }
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 32
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withTools() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(
                model: model,
                tools: [weatherTool],
                instructions: "You are a helpful assistant. Use available tools when needed."
            )

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(!toolOutput.id.isEmpty)
                #expect(toolOutput.toolName == weatherTool.name)
                foundToolOutput = true
            }
            #expect(foundToolOutput)
            #expect(response.usage.input.totalTokenCount > 0)
            #expect(response.usage.output.totalTokenCount > 0)
            #expect(session.usage == response.usage)

            let calls = await weatherTool.calls
            #expect(calls.count >= 1)
            if let first = calls.first {
                #expect(first.arguments.city.contains("San Francisco"))
            }
        }

        @Test func streamingWithTools() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(
                model: model,
                tools: [weatherTool],
                instructions: "You are a helpful assistant. Use available tools when needed."
            )

            let stream = session.streamResponse(to: "How's the weather in San Francisco?")

            // Iterate the stream, keeping the last snapshot as the final state.
            var snapshotCount = 0
            var lastSnapshot: LanguageModelSession.ResponseStream<String>.Snapshot?
            for try await snapshot in stream {
                snapshotCount += 1
                lastSnapshot = snapshot
            }

            // The stream yielded incremental snapshots and produced text.
            #expect(snapshotCount >= 1)
            #expect(!(lastSnapshot?.content.isEmpty ?? true))

            // The tool actually executed.
            let calls = await weatherTool.calls
            #expect(calls.count >= 1)
            if let first = calls.first {
                #expect(first.arguments.city.contains("San Francisco"))
            }

            // Tool activity surfaces through the stream's transcript entries.
            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in lastSnapshot?.transcriptEntries ?? [] {
                #expect(!toolOutput.id.isEmpty)
                #expect(toolOutput.toolName == weatherTool.name)
                foundToolOutput = true
            }
            #expect(foundToolOutput)
            #expect((lastSnapshot?.usage.input.totalTokenCount ?? 0) > 0)
            #expect((lastSnapshot?.usage.output.totalTokenCount ?? 0) > 0)
            #expect(session.usage == lastSnapshot?.usage)
        }

        @Test func multimodalWithImageURL() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Describe this image")),
                        .image(.init(url: testImageURL)),
                    ])
                )
            ])
            let session = LanguageModelSession(model: visionModel, transcript: transcript)
            var options = GenerationOptions()
            var mlxOptions = MLXLanguageModel.CustomGenerationOptions.default
            mlxOptions.userInputProcessing = .resize(to: CGSize(width: 512, height: 512))
            options[custom: MLXLanguageModel.self] = mlxOptions
            let response = try await session.respond(to: "", options: options)
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageData() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Describe this image")),
                        .image(.init(data: testImageData, mimeType: "image/png")),
                    ])
                )
            ])
            let session = LanguageModelSession(model: visionModel, transcript: transcript)
            var options = GenerationOptions()
            var mlxOptions = MLXLanguageModel.CustomGenerationOptions.default
            mlxOptions.userInputProcessing = .resize(to: CGSize(width: 512, height: 512))
            options[custom: MLXLanguageModel.self] = mlxOptions
            let response = try await session.respond(to: "", options: options)
            #expect(!response.content.isEmpty)
        }

        @Test func structuredGenerationSimpleString() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a greeting message that says hello",
                generating: SimpleString.self
            )
            #expect(!response.content.message.isEmpty)
        }

        @Test func structuredGenerationSimpleInt() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a count value of 42",
                generating: SimpleInt.self
            )
            #expect(response.content.count >= 0)
        }

        @Test func structuredGenerationSimpleDouble() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a temperature value of 72.5 degrees",
                generating: SimpleDouble.self
            )
            #expect(!response.content.temperature.isNaN)
        }

        @Test func structuredGenerationSimpleBool() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a boolean value: true",
                generating: SimpleBool.self
            )
            #expect(response.content.value == true)
            let jsonData = response.rawContent.jsonString.data(using: .utf8)
            #expect(jsonData != nil)
            if let jsonData {
                let json = try JSONSerialization.jsonObject(with: jsonData)
                let dictionary = json as? [String: Any]
                let boolValue = dictionary?["value"] as? Bool
                #expect(boolValue != nil)
            }
        }

        @Test func structuredGenerationOptionalFields() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a person named Alex with nickname 'Lex'. Nickname may be omitted if unsure.",
                generating: OptionalFields.self
            )
            #expect(!response.content.name.isEmpty)
            if let nickname = response.content.nickname {
                #expect(!nickname.isEmpty)
            }
        }

        @Test func structuredGenerationEnum() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a high priority value",
                generating: Priority.self
            )
            #expect([Priority.low, Priority.medium, Priority.high].contains(response.content))
        }

        @Test func withAdditionalContext() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 32
            )
            var custom = MLXLanguageModel.CustomGenerationOptions.default
            custom.additionalContext = [
                "user_name": JSONValue.string("Alice"),
                "turn_count": JSONValue.int(3),
                "verbose": JSONValue.bool(true),
            ]
            options[custom: MLXLanguageModel.self] = custom

            let response = try await session.respond(
                to: "Say hello",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func unavailableForNonexistentModel() async {
            let model = MLXLanguageModel(modelId: "mlx-community/does-not-exist-anylanguagemodel-test")
            await model.removeFromCache()
            #expect(model.availability == .unavailable(.notLoaded))
            #expect(model.isAvailable == false)

            let session = LanguageModelSession(model: model)
            await #expect(throws: Error.self) {
                _ = try await session.respond(to: "Hello")
            }

            switch model.availability {
            case .unavailable(.failedToLoad(let description)):
                #expect(!description.isEmpty)
            default:
                Issue.record("Expected model availability to report failedToLoad after failed request")
            }
            #expect(model.isAvailable == false)
        }

        @Test func removeAllFromCacheThenRespond() async throws {
            await MLXLanguageModel.removeAllFromCache()
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(to: "Say hello after cache clear")
            #expect(!response.content.isEmpty)
        }
    }
#endif  // MLX
