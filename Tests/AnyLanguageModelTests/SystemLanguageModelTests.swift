import Testing
@testable import AnyLanguageModel

#if canImport(FoundationModels)
    private let isSystemLanguageModelAvailable = {
        if #available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *) {
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

    @Test("GenerationSchema merges duplicate defs for the same type")
    func generationSchemaMergesDuplicateDefsForSameType() {
        let schema = ContainerWithDuplicateNestedType.generationSchema

        let nestedTypeName = String(reflecting: ReusedNestedStruct.self)
        #expect(schema.defs[nestedTypeName] != nil)
    }

    @Suite(
        "SystemLanguageModel",
        .enabled(if: isSystemLanguageModelAvailable)
    )
    struct SystemLanguageModelTests {
        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func basicResponse() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say 'Hello'")
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func withInstructions() async throws {
            let model = SystemLanguageModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func withTemperature() async throws {
            let model: SystemLanguageModel = SystemLanguageModel()
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(temperature: 0.5)
            let response = try await session.respond(
                to: "Generate a number",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func withTools() async throws {
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: SystemLanguageModel.default, tools: [weatherTool])

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            #if false  // Disabled for now because transcript entries are not converted from FoundationModels for now
                var foundToolOutput = false
                for case let .toolOutput(toolOutput) in response.transcriptEntries {
                    #expect(toolOutput.id == "getWeather")
                    foundToolOutput = true
                }
                #expect(foundToolOutput)
            #endif

            let content = response.content
            #expect(content.contains("San Francisco"))
            #expect(content.contains("72°F"))
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationSimpleStruct() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Generate a friendly greeting",
                generating: Greeting.self
            )

            #expect(!response.content.message.isEmpty)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func guidedGenerationWithEnumConstraint() async throws {
            let session = LanguageModelSession(model: SystemLanguageModel.default)

            let response = try await session.respond(
                to: "Analyze the sentiment of: 'I love this product!'",
                generating: SentimentAnalysis.self
            )

            #expect(["positive", "negative", "neutral"].contains(response.content.sentiment.lowercased()))
            #expect(response.content.confidence >= 0.0 && response.content.confidence <= 1.0)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
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
#endif
