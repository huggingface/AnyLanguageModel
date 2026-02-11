import Foundation
import Testing

@testable import AnyLanguageModel

#if CoreML
    import Hub
    import CoreML

    private let shouldRunCoreMLTests: Bool = {
        // Enable when explicitly requested via environment variable
        if ProcessInfo.processInfo.environment["ENABLE_COREML_TESTS"] != nil {
            return true
        }

        // Skip in CI environments
        if ProcessInfo.processInfo.environment["CI"] != nil {
            return false
        }

        return true
    }()

    @Suite("CoreMLLanguageModel", .enabled(if: shouldRunCoreMLTests), .serialized)
    struct CoreMLLanguageModelTests {
        let modelId = "apple/mistral-coreml"
        let modelPackageName = "StatefulMistral7BInstructInt4.mlpackage"

        @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        private static let modelTask = Task {
            let hasToken = ProcessInfo.processInfo.environment["HF_TOKEN"] != nil
            let hubApi = HubApi(useOfflineMode: !hasToken)
            let repoURL = try await hubApi.snapshot(
                from: Hub.Repo(id: "apple/mistral-coreml", type: .models),
                matching: "*Int4.mlpackage/**"
            ) { progress in
                print("Download progress: \(Int(progress.fractionCompleted * 100))%")
            }

            let modelURL = repoURL.appending(component: "StatefulMistral7BInstructInt4.mlpackage")
            let compiledURL: URL
            if modelURL.pathExtension == "mlmodelc" {
                compiledURL = modelURL
            } else {
                compiledURL = try await MLModel.compileModel(at: modelURL)
            }
            return try await CoreMLLanguageModel(url: compiledURL)
        }

        @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func getModel() async throws -> CoreMLLanguageModel {
            try await Self.modelTask.value
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func basicResponse() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withGenerationOptions() async throws {
            let model = try await getModel()
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

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func withSamplingStrategies() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            // Test greedy sampling
            let greedyOptions = GenerationOptions(sampling: .greedy)
            let greedyResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: greedyOptions
            )
            #expect(!greedyResponse.content.isEmpty)

            // Test top-k sampling
            let topKOptions = GenerationOptions(sampling: .random(top: 10))
            let topKResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: topKOptions
            )
            #expect(!topKResponse.content.isEmpty)

            // Test nucleus sampling
            let nucleusOptions = GenerationOptions(sampling: .random(probabilityThreshold: 0.9))
            let nucleusResponse = try await session.respond(
                to: "Complete this sentence: The sky is",
                options: nucleusOptions
            )
            #expect(!nucleusResponse.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func temperatureVariations() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            // Test low temperature (more deterministic)
            let lowTempOptions = GenerationOptions(temperature: 0.1)
            let lowTempResponse = try await session.respond(
                to: "Write a short story about a cat",
                options: lowTempOptions
            )
            #expect(!lowTempResponse.content.isEmpty)

            // Test high temperature (more creative)
            let highTempOptions = GenerationOptions(temperature: 0.9)
            let highTempResponse = try await session.respond(
                to: "Write a short story about a cat",
                options: highTempOptions
            )
            #expect(!highTempResponse.content.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func maxTokensLimit() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(maximumResponseTokens: 5)
            let response = try await session.respond(
                to: "Write a long story about space exploration",
                options: options
            )
            #expect(!response.content.isEmpty)
            // Note: We can't easily test token count without access to the tokenizer
            // but we can verify the response is not empty
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func multimodal_rejectsImageURL() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)
            do {
                _ = try await session.respond(
                    to: "Describe this image",
                    image: .init(url: testImageURL)
                )
                Issue.record("Expected error when image segments are present")
            } catch {
                // CoreMLUnsupportedFeatureError is a private struct, so we just check that an error is thrown
                #expect(Bool(true))
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func multimodal_rejectsImageData() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(model: model)
            do {
                _ = try await session.respond(
                    to: "Describe this image",
                    image: .init(data: testImageData, mimeType: "image/jpeg")
                )
                Issue.record("Expected error when image segments are present")
            } catch {
                // CoreMLUnsupportedFeatureError is a private struct, so we just check that an error is thrown
                #expect(Bool(true))
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleString() async throws {
            let model = try await getModel()
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

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleInt() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a count value of 42",
                generating: SimpleInt.self
            )
            #expect(response.content.count >= 0)
            let jsonData = response.rawContent.jsonString.data(using: .utf8)
            #expect(jsonData != nil)
            if let jsonData {
                let json = try JSONSerialization.jsonObject(with: jsonData)
                let dictionary = json as? [String: Any]
                #expect(dictionary != nil)
                if let dictionary {
                    let countValue = dictionary["count"] as? NSNumber
                    #expect(countValue?.intValue != nil)
                }
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleBool() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a boolean value: true",
                generating: SimpleBool.self
            )
            let jsonData = response.rawContent.jsonString.data(using: .utf8)
            #expect(jsonData != nil)
            if let jsonData {
                let json = try JSONSerialization.jsonObject(with: jsonData)
                let dictionary = json as? [String: Any]
                #expect(dictionary != nil)
                if let dictionary {
                    let boolValue = dictionary["value"] as? Bool
                    #expect(boolValue != nil)
                }
            }
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleDouble() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a temperature value of 72.5 degrees",
                generating: SimpleDouble.self
            )
            #expect(!response.content.temperature.isNaN)
            #expect(response.content.temperature.isFinite)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationOptionalFields() async throws {
            let model = try await getModel()
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

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationEnum() async throws {
            let model = try await getModel()
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

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationSimpleArray() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a list of 3 color names: red, green, blue",
                generating: SimpleArray.self
            )
            #expect(!response.content.colors.isEmpty)
        }

        @Test @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
        func structuredGenerationNestedStruct() async throws {
            let model = try await getModel()
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a person named John, age 25, living at 123 Main St, Springfield, 12345",
                generating: StructuredPerson.self
            )
            #expect(!response.content.name.isEmpty)
            #expect(response.content.age >= 0)
            #expect(!response.content.address.street.isEmpty)
            #expect(!response.content.address.city.isEmpty)
        }
    }
#endif  // CoreML
