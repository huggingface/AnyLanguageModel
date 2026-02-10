import Foundation
import Testing

@testable import AnyLanguageModel

private let openResponsesAPIKey: String? = ProcessInfo.processInfo.environment["OPEN_RESPONSES_API_KEY"]
private let openResponsesBaseURL: URL? = ProcessInfo.processInfo.environment["OPEN_RESPONSES_BASE_URL"].flatMap {
    URL(string: $0)
}

@Suite("OpenResponsesLanguageModel")
struct OpenResponsesLanguageModelTests {
    @Test func customHost() throws {
        let customURL = URL(string: "https://example.com")!
        let model = OpenResponsesLanguageModel(baseURL: customURL, apiKey: "test", model: "test-model")
        #expect(model.baseURL.absoluteString.hasSuffix("/"))
    }

    @Test func modelParameter() throws {
        let baseURL = URL(string: "https://api.example.com/v1/")!
        let model = OpenResponsesLanguageModel(baseURL: baseURL, apiKey: "test-key", model: "my-model")
        #expect(model.model == "my-model")
    }

    @Suite(
        "OpenResponsesLanguageModel API",
        .enabled(if: openResponsesAPIKey?.isEmpty == false && openResponsesBaseURL != nil)
    )
    struct APITests {
        private let apiKey = openResponsesAPIKey!
        private var baseURL: URL { openResponsesBaseURL! }

        private var model: OpenResponsesLanguageModel {
            OpenResponsesLanguageModel(
                baseURL: baseURL,
                apiKey: apiKey,
                model: "gpt-4o-mini"
            )
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )
            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)
            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []
            for try await response in stream {
                chunks.append(response.content)
            }
            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)
            let stream = session.streamResponse(to: "Say 'Hello' slowly")
            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }
            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )
            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)
            var options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )
            options[custom: OpenResponsesLanguageModel.self] = .init(
                extraBody: ["user": .string("test-user-id")]
            )
            let response = try await session.respond(
                to: "Say hello",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageURL() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Describe this image",
                image: .init(url: testImageURL)
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageData() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Describe this image",
                image: .init(data: testImageData, mimeType: "image/png")
            )
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)
            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)
            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(secondResponse.content.contains("color"))
        }

        @Test func withTools() async throws {
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: model, tools: [weatherTool])
            let response = try await session.respond(to: "How's the weather in San Francisco?")
            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.toolName == "getWeather")
                foundToolOutput = true
            }
            #expect(foundToolOutput)
        }

        @Suite("Structured Output")
        struct StructuredOutputTests {
            @Generable
            struct Person {
                @Guide(description: "The person's full name")
                var name: String

                @Guide(description: "The person's age in years")
                var age: Int

                @Guide(description: "The person's email address")
                var email: String?
            }

            @Generable
            struct Book {
                @Guide(description: "The book's title")
                var title: String

                @Guide(description: "The book's author")
                var author: String

                @Guide(description: "The publication year")
                var year: Int
            }

            private var model: OpenResponsesLanguageModel {
                OpenResponsesLanguageModel(
                    baseURL: openResponsesBaseURL!,
                    apiKey: openResponsesAPIKey!,
                    model: "gpt-4o-mini"
                )
            }

            @Test func basicStructuredOutput() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a person named John Doe, age 30, email john@example.com",
                    generating: Person.self
                )
                #expect(!response.content.name.isEmpty)
                #expect(response.content.name.contains("John") || response.content.name.contains("Doe"))
                #expect(response.content.age > 0)
                #expect(response.content.age <= 100)
                #expect(response.content.email != nil)
            }

            @Test func structuredOutputWithOptionalField() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a person named Jane Smith, age 25, with no email",
                    generating: Person.self
                )
                #expect(!response.content.name.isEmpty)
                #expect(response.content.email == nil || response.content.email?.isEmpty == true)
            }

            @Test func structuredOutputWithNestedTypes() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a book titled 'The Swift Programming Language' by 'Apple Inc.' published in 2024",
                    generating: Book.self
                )
                #expect(!response.content.title.isEmpty)
                #expect(!response.content.author.isEmpty)
                #expect(response.content.year >= 2020)
            }

            @Test func streamingStructuredOutput() async throws {
                let session = LanguageModelSession(model: model)
                let stream = session.streamResponse(
                    to: "Generate a person named Alice, age 28, email alice@example.com",
                    generating: Person.self
                )
                var snapshots: [LanguageModelSession.ResponseStream<Person>.Snapshot] = []
                for try await snapshot in stream {
                    snapshots.append(snapshot)
                }
                #expect(!snapshots.isEmpty)
                let finalSnapshot = snapshots.last!
                #expect((finalSnapshot.content.name?.isEmpty ?? true) == false)
                #expect((finalSnapshot.content.age ?? 0) > 0)
            }
        }
    }
}
