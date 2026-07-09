import Foundation
import Testing

@testable import AnyLanguageModel

private let anthropicAPIKey: String? = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]

@Generable
private struct AnthropicStructuredForecast {
    var summary: String
    var temperatureCelsius: Int
}

@Suite("AnthropicLanguageModel", .serialized, .enabled(if: anthropicAPIKey?.isEmpty == false))
struct AnthropicLanguageModelTests {
    let model = AnthropicLanguageModel(
        apiKey: anthropicAPIKey!,
        model: "claude-sonnet-4-5"
    )

    @Test func customHost() throws {
        let customURL = URL(string: "https://example.com")!
        let model = AnthropicLanguageModel(baseURL: customURL, apiKey: "test", model: "test-model")
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

    @Test func streamingStructured() async throws {
        let session = LanguageModelSession(model: model)

        let stream = session.streamResponse(
            to: "Provide a short weather forecast summary and a celsius temperature.",
            generating: AnthropicStructuredForecast.self
        )

        var snapshots: [LanguageModelSession.ResponseStream<AnthropicStructuredForecast>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
        }

        #expect(!snapshots.isEmpty)
        #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        #expect(!(snapshots.last!.content.summary ?? "").isEmpty)
    }
    
    @Test func streamingTranscript() async throws {
        let session = LanguageModelSession(model: model)
        
        let stream = session.streamResponse(to: "Say 'Hello' slowly")
        
        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
            
            // Make sure the snapshot is also in the transcript.
            let hasTranscriptEntry = session.transcript.contains(where: { entry in
                switch entry {
                case .response(let response):
                    return response.segments.contains { segment in
                        switch segment {
                        case .text(let text):
                            return text.content.contains(snapshot.content)
                        default:
                            return false
                        }
                    }
                default:
                    return false
                }
            })
            #expect(hasTranscriptEntry, "Expected the string snapshot to also appear in the transcript during streaming.")
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

    @Test func structuredResponse() async throws {
        let session = LanguageModelSession(model: model)

        let response = try await session.respond(
            to: "Summarize the weather with a short summary and a celsius temperature.",
            generating: AnthropicStructuredForecast.self
        )

        #expect(!response.content.summary.isEmpty)
        #expect(response.rawContent.jsonString.contains("summary"))
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
            #expect(!toolOutput.id.isEmpty)
            #expect(toolOutput.toolName == "getWeather")
            foundToolOutput = true
        }
        #expect(foundToolOutput)
    }


    @Test func streamWithTools() async throws {
        let weatherTool = WeatherTool()
        let session = LanguageModelSession(model: model, tools: [weatherTool])
        
        let stream = session.streamResponse(to: "How's the weather in San Francisco?")
        
        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        
        var toolAppearedInTranscript: Bool = false
        var toolResponseAppearedInTranscript: Bool = false
        
        for try await snapshot in stream {
            snapshots.append(snapshot)
            
            for entry in session.transcript {
                switch entry {
                case .toolCalls:
                    toolAppearedInTranscript = true
                case .toolOutput:
                    toolResponseAppearedInTranscript = true
                default: break
                }
            }
        }
        
        #expect(toolAppearedInTranscript, "Expected a tool call to appear in the transcript during streaming.")
        #expect(toolResponseAppearedInTranscript, "Expected a tool output to appear in the transcript during streaming.")
        
        var foundToolOutput = false
        for case let .toolOutput(toolOutput) in session.transcript {
            #expect(!toolOutput.id.isEmpty)
            #expect(toolOutput.toolName == "getWeather")
            foundToolOutput = true
        }
        #expect(foundToolOutput, "Expected the 'getWeather' tool to exist in the final transcript.")
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
}
