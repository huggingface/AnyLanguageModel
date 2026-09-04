import Foundation
import Testing

@testable import AnyLanguageModel

@Generable
private struct OllamaStructuredForecast {
    var summary: String
    var temperatureCelsius: Int
}

@Suite(
    "OllamaLanguageModel",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["CI"] == nil)
)
struct OllamaLanguageModelTests {
    let model = OllamaLanguageModel(model: "qwen3:8b")

    @Test func customHost() {
        let customURL = URL(string: "http://example.com")!
        let model = OllamaLanguageModel(baseURL: customURL, model: "custom")
        #expect(model.baseURL.absoluteString.hasSuffix("/"))
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

    @Test func structuredResponse() async throws {
        let session = LanguageModelSession(model: model)

        let response = try await session.respond(
            to: "Summarize the weather with a short summary and a celsius temperature.",
            generating: OllamaStructuredForecast.self
        )

        #expect(!response.content.summary.isEmpty)
        #expect(response.rawContent.jsonString.contains("summary"))
    }

    @Test func streamingStructured() async throws {
        let session = LanguageModelSession(model: model)

        let stream = session.streamResponse(
            to: "Provide a short weather forecast summary and a celsius temperature.",
            generating: OllamaStructuredForecast.self
        )

        var snapshots: [LanguageModelSession.ResponseStream<OllamaStructuredForecast>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
        }

        #expect(!snapshots.isEmpty)
        #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        #expect(!(snapshots.last!.content.summary ?? "").isEmpty)
    }

    @Test func withGenerationOptions() async throws {
        let session = LanguageModelSession(model: model)

        let options = GenerationOptions(
            temperature: 0.7,
        )

        let response = try await session.respond(
            to: "Tell me a fact",
            options: options
        )
        #expect(!response.content.isEmpty)
    }

    @Test func conversationContext() async throws {
        let session = LanguageModelSession(model: model)

        let firstResponse = try await session.respond(to: "My favorite color is blue")
        #expect(!firstResponse.content.isEmpty)

        let secondResponse = try await session.respond(to: "What did I just tell you?")
        #expect(!secondResponse.content.isEmpty)
    }

    @Test func withTools() async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: model, tools: [weatherTool])

        let response = try await session.respond(to: "How's the weather in San Francisco?")

        var foundToolOutput = false
        for case let .toolOutput(toolOutput) in response.transcriptEntries {
            #expect(!toolOutput.id.isEmpty)
            #expect(toolOutput.toolName == weatherTool.name)
            foundToolOutput = true
        }
        #expect(foundToolOutput)

        let calls = await weatherTool.calls
        #expect(calls.count == 1)
        #expect(calls.first?.arguments.city == "San Francisco")

        if case .success(let output) = calls.first?.result {
            #expect(output.contains("San Francisco"))
        } else {
            Issue.record("Expected successful tool call")
        }
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
        let session = LanguageModelSession(model: model, transcript: transcript)
        let response = try await session.respond(to: "")
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
        let session = LanguageModelSession(model: model, transcript: transcript)
        let response = try await session.respond(to: "")
        #expect(!response.content.isEmpty)
    }
}

@Suite("Ollama chat request encoding")
struct OllamaChatRequestEncodingTests {
    @Test func attachesImagesToTheUserMessage() throws {
        let base64Image = Data([0xFF, 0xD8, 0xFF]).base64EncodedString()
        let params = try createChatParams(
            model: "llava",
            messages: [
                OllamaMessage(
                    role: .user,
                    content: "What is in this image?",
                    images: [base64Image]
                )
            ],
            tools: nil,
            options: nil,
            stream: false,
            format: nil
        )

        #expect(params["images"] == nil)

        guard case .array(let messages)? = params["messages"],
            case .object(let message)? = messages.first
        else {
            Issue.record("Expected messages to encode as an array of objects")
            return
        }
        #expect(message["images"] == .array([.string(base64Image)]))
    }

    @Test func omitsImagesForTextOnlyMessages() throws {
        let params = try createChatParams(
            model: "llama3.2",
            messages: [OllamaMessage(role: .user, content: "Hello")],
            tools: nil,
            options: nil,
            stream: false,
            format: nil
        )

        #expect(params["images"] == nil)

        guard case .array(let messages)? = params["messages"],
            case .object(let message)? = messages.first
        else {
            Issue.record("Expected messages to encode as an array of objects")
            return
        }
        #expect(message["images"] == nil)
    }
}

@Suite("Ollama top-level chat parameters")
struct OllamaTopLevelChatParametersTests {
    @Test func routesThinkToTheTopLevelOfTheRequest() throws {
        var options = GenerationOptions()
        options[custom: OllamaLanguageModel.self] = [
            "think": .bool(true),
            "repeat_penalty": .double(1.2),
        ]

        let params = try createChatParams(
            model: "qwen3:8b",
            messages: [OllamaMessage(role: .user, content: "Hello")],
            tools: nil,
            options: convertOptions(options),
            stream: false,
            format: nil,
            parameters: extractTopLevelChatParameters(options)
        )

        #expect(params["think"] == .bool(true))

        guard case .object(let requestOptions)? = params["options"] else {
            Issue.record("Expected options to encode as an object")
            return
        }
        #expect(requestOptions["think"] == nil)
        #expect(requestOptions["repeat_penalty"] == .double(1.2))
    }

    @Test func topLevelParametersDoNotOverrideReservedKeys() throws {
        let params = try createChatParams(
            model: "gpt-oss:20b",
            messages: [OllamaMessage(role: .user, content: "Hello")],
            tools: nil,
            options: nil,
            stream: false,
            format: nil,
            parameters: ["model": .string("injected"), "think": .string("high")]
        )

        #expect(params["model"] == .string("gpt-oss:20b"))
        #expect(params["think"] == .string("high"))
    }
}
