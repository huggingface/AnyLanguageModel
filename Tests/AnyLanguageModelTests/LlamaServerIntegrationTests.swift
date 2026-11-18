import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// These tests run the remote backends against a local llama-server,
// the HTTP server in llama.cpp (https://github.com/ggml-org/llama.cpp).
// They run only when `LLAMA_SERVER_BASE_URL` names a running server,
// such as `http://127.0.0.1:8080`.
// Set `LLAMA_SERVER_VISION_BASE_URL` to a server with a vision model
// to run the image tests.
// The README's llama-server section lists the models and flags they were run with.

private let llamaServerBaseURL = environmentURL("LLAMA_SERVER_BASE_URL")
private let llamaServerVisionBaseURL = environmentURL("LLAMA_SERVER_VISION_BASE_URL")

private func environmentURL(_ name: String) -> URL? {
    guard let value = ProcessInfo.processInfo.environment[name], !value.isEmpty else { return nil }
    return URL(string: value)
}

/// A backend in this package that a llama-server endpoint serves.
enum LlamaServerBackend: String, CaseIterable, CustomTestStringConvertible, Sendable {
    /// `OpenAILanguageModel` with `/v1/chat/completions`.
    case chatCompletions
    /// `OpenAILanguageModel` with `/v1/responses`.
    case responses
    /// `OpenResponsesLanguageModel` with `/v1/responses`.
    case openResponses
    /// `AnthropicLanguageModel` with `/v1/messages`.
    case anthropicMessages

    var testDescription: String { rawValue }

    /// A model for the server at `baseURL`.
    ///
    /// A server with one model answers with it whatever the request's `model` is,
    /// so the name here only shows in the server's log.
    func model(at baseURL: URL) -> any LanguageModel {
        let v1 = baseURL.appendingPathComponent("v1/")
        switch self {
        case .chatCompletions:
            return OpenAILanguageModel(baseURL: v1, apiKey: "", model: "llama-server", apiVariant: .chatCompletions)
        case .responses:
            return OpenAILanguageModel(baseURL: v1, apiKey: "", model: "llama-server", apiVariant: .responses)
        case .openResponses:
            return OpenResponsesLanguageModel(baseURL: v1, apiKey: "", model: "llama-server")
        case .anthropicMessages:
            // The Messages API path is `v1/messages` from the server's root.
            return AnthropicLanguageModel(baseURL: baseURL, apiKey: "", model: "llama-server")
        }
    }

    /// Whether the backend records the model's reasoning in the transcript.
    ///
    /// The Responses backends don't read `reasoning` output items.
    var recordsReasoning: Bool {
        switch self {
        case .chatCompletions, .anthropicMessages: true
        case .responses, .openResponses: false
        }
    }

    /// Whether the backend's API has a stop-sequence parameter.
    ///
    /// The Responses API has none.
    var hasStopSequences: Bool {
        switch self {
        case .chatCompletions, .anthropicMessages: true
        case .responses, .openResponses: false
        }
    }

    /// Options with stop sequences for the backend, without thinking.
    ///
    /// llama-server matches stop sequences in the reasoning too,
    /// so a model that thinks can stop before it answers.
    /// Chat Completions turns thinking off with `reasoning_effort`;
    /// the Messages endpoint ignores `thinking.type: "disabled"`,
    /// so it gets llama-server's `chat_template_kwargs`.
    func options(stoppingAt sequences: [String]) -> GenerationOptions {
        var options = GenerationOptions()
        switch self {
        case .chatCompletions:
            options[custom: OpenAILanguageModel.self] = .init(
                stopSequences: sequences,
                reasoningEffort: OpenAILanguageModel.CustomGenerationOptions.ReasoningEffort.none
            )
        case .anthropicMessages:
            options[custom: AnthropicLanguageModel.self] = .init(
                stopSequences: sequences,
                extraBody: ["chat_template_kwargs": ["enable_thinking": false]]
            )
        case .responses, .openResponses:
            break
        }
        return options
    }

    /// Options for tool calls one at a time, at temperature 0.
    ///
    /// A model can call a tool that needs another's result in the same round,
    /// with a placeholder argument, unless the request turns parallel calls off.
    /// The Messages API has no such option at the top level.
    var sequentialToolOptions: GenerationOptions {
        var options = GenerationOptions(temperature: 0)
        switch self {
        case .chatCompletions, .responses:
            options[custom: OpenAILanguageModel.self] = .init(parallelToolCalls: false)
        case .openResponses:
            options[custom: OpenResponsesLanguageModel.self] = .init(parallelToolCalls: false)
        case .anthropicMessages:
            break
        }
        return options
    }

    /// Whether llama-server enforces this backend's response schema
    /// while the request also has tools.
    ///
    /// With `tools` and `response_format` or `text.format`,
    /// llama-server's grammar allows only the response format,
    /// so the model can't call a tool.
    var callsToolsWithResponseSchema: Bool {
        self == .anthropicMessages
    }

    /// Whether llama-server enforces this backend's response schema.
    ///
    /// The Messages endpoint ignores `output_config.format`.
    var enforcesResponseSchema: Bool {
        self != .anthropicMessages
    }

    /// Options that ask a streamed response for its usage.
    ///
    /// `OpenAILanguageModel` sends `stream_options.include_usage` only to OpenAI,
    /// and llama-server reports a stream's usage only when asked.
    var streamingUsageOptions: GenerationOptions {
        var options = GenerationOptions()
        if self == .chatCompletions {
            options[custom: OpenAILanguageModel.self] = .init(
                extraBody: ["stream_options": ["include_usage": true]]
            )
        }
        return options
    }
}

@Generable
private struct LlamaServerPerson {
    @Guide(description: "The person's first name")
    var name: String

    @Guide(description: "The person's age in years")
    var age: Int
}

/// Looks up a fictional country's capital,
/// so a model must call it to learn the answer.
private struct CapitalTool: Tool {
    let name = "getCapital"
    let description = "Look up the capital city of a country"

    @Generable
    struct Arguments {
        @Guide(description: "The country to look up")
        var country: String
    }

    func call(arguments: Arguments) async throws -> String {
        "The capital of \(arguments.country) is Zarnow."
    }
}

/// A 64 × 64 PNG image of solid red.
private let redImageData = Data(
    base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PQQkAAAgAsetfWiP4FgYrsKZeS0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDgsqnc8OJg6Ln3AAAAAElFTkSuQmCC"
)!

@Suite(
    "llama-server",
    .serialized,
    .enabled(if: llamaServerBaseURL != nil, "Set LLAMA_SERVER_BASE_URL to a running llama-server")
)
struct LlamaServerIntegrationTests {
    private let baseURL = llamaServerBaseURL!

    // MARK: Text

    @Test(arguments: LlamaServerBackend.allCases)
    func textResponse(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.respond(to: "Reply with only the word: hello")
        attachResponse(response)

        #expect(response.content.localizedCaseInsensitiveContains("hello"))
        #expect(response.usage.input.totalTokenCount > 0)
        #expect(response.usage.output.totalTokenCount > 0)
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func streamedResponse(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let stream = session.streamResponse(
            to: "Count from 1 to 5, separated by commas.",
            options: backend.streamingUsageOptions
        )

        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
        }

        #expect(snapshots.count > 1)
        let last = try #require(snapshots.last)
        #expect(try last.rawContent.value(String.self).contains("3"))
        #expect(last.usage.input.totalTokenCount > 0)
        #expect(last.usage.output.totalTokenCount > 0)
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func instructions(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(
            model: backend.model(at: baseURL),
            instructions: "Whatever the user asks, reply with only the word: banana"
        )
        let response = try await session.respond(to: "What is the capital of France?")
        attachResponse(response)

        #expect(response.content.localizedCaseInsensitiveContains("banana"))
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func multiTurnTranscript(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let first = try await session.respond(to: "My favorite color is teal. Reply with only: OK")
        attachResponse(first, "first")

        let second = try await session.respond(to: "What is my favorite color? Reply with one word.")
        attachResponse(second, "second")

        #expect(second.content.localizedCaseInsensitiveContains("teal"))
    }

    // MARK: Structured output

    @Test(arguments: LlamaServerBackend.allCases)
    func structuredOutput(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.respond(
            to: "Ada is 36 years old. Describe her.",
            generating: LlamaServerPerson.self
        )
        attachResponse(response)

        #expect(response.content.name.contains("Ada"))
        #expect(response.content.age == 36)
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func structuredOutputWithoutSchemaInPrompt(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        try await withKnownIssue(
            "llama-server's Messages endpoint ignores output_config.format",
            isIntermittent: true
        ) {
            let response = try await session.respond(
                to: "Say hello to Ada, who is 36.",
                generating: LlamaServerPerson.self,
                includeSchemaInPrompt: false
            )
            attachResponse(response)

            #expect(response.content.name.contains("Ada"))
            #expect(response.content.age == 36)
        } when: {
            !backend.enforcesResponseSchema
        }
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func streamedStructuredOutput(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let stream = session.streamResponse(
            to: "Ada is 36 years old. Describe her.",
            generating: LlamaServerPerson.self
        )

        var snapshots: [LanguageModelSession.ResponseStream<LlamaServerPerson>.Snapshot] = []
        for try await snapshot in stream {
            snapshots.append(snapshot)
        }

        let last = try #require(snapshots.last)
        #expect(last.content.name?.contains("Ada") == true)
        #expect(last.content.age == 36)
    }

    // MARK: Tools

    @Test(arguments: LlamaServerBackend.allCases)
    func toolCall(backend: LlamaServerBackend) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: backend.model(at: baseURL), tools: [weatherTool])
        let response = try await session.respond(
            to: "Use the getWeather tool to find the weather in Paris, then tell me the temperature."
        )
        attachResponse(response)

        let calls = await weatherTool.calls
        #expect(calls.count == 1)
        #expect(calls.first?.arguments.city.contains("Paris") == true)
        #expect(response.transcriptEntries.contains { if case .toolCalls = $0 { true } else { false } })
        #expect(response.transcriptEntries.contains { if case .toolOutput = $0 { true } else { false } })
        #expect(response.content.contains("72") || response.content.contains("23"))
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func severalToolRounds(backend: LlamaServerBackend) async throws {
        let capitalTool = spy(on: CapitalTool())
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(
            model: backend.model(at: baseURL),
            tools: [capitalTool, weatherTool]
        )
        // The Messages endpoint has no way to turn parallel calls off.
        // Qwen3 then sometimes calls getWeather in the first round with a guessed city,
        // or repeats parallel calls until it runs out of tokens
        // and the last call has no arguments, which throws when the tool runs.
        try await withKnownIssue(
            "llama-server's Messages endpoint can't turn off parallel tool calls",
            isIntermittent: true
        ) {
            let response = try await session.respond(
                to: """
                    Use getCapital to find the capital of Freedonia. \
                    Then use getWeather for that capital. \
                    Then tell me the capital and its weather.
                    """,
                options: backend.sequentialToolOptions
            )
            attachResponse(response)

            // What the package guarantees whenever the model calls a tool:
            // each call runs once, and its output follows the calls in order.
            let capitalCalls = await capitalTool.calls
            let weatherCalls = await weatherTool.calls
            #expect(capitalCalls.count >= 1)
            let rounds = expectOutputsFollowCalls(in: response.transcriptEntries)
            #expect(rounds.map(\.count).reduce(0, +) == capitalCalls.count + weatherCalls.count)

            // What depends on the model's choices.
            // Even at temperature 0 with parallel calls off,
            // a small model sometimes stops after the first round,
            // writes the next call inside its unfinished thinking
            // (where llama-server leaves it as reasoning),
            // or calls getWeather with a guessed city.
            withKnownIssue("The model sometimes takes fewer tool rounds", isIntermittent: true) {
                #expect(weatherCalls.contains { $0.arguments.city.contains("Zarnow") })
                #expect(rounds.count >= 2)
                #expect(response.content.contains("Zarnow"))
            }
        } when: {
            backend == .anthropicMessages
        }
    }

    /// Expects each `toolCalls` entry in `entries`
    /// to be followed by one output for each of its calls, in the same order,
    /// and returns the rounds of calls.
    private func expectOutputsFollowCalls(
        in entries: ArraySlice<Transcript.Entry>,
        sourceLocation: SourceLocation = #_sourceLocation
    ) -> [Transcript.ToolCalls] {
        let entries = Array(entries)
        var rounds: [Transcript.ToolCalls] = []
        for (index, entry) in entries.enumerated() {
            guard case .toolCalls(let calls) = entry else { continue }
            rounds.append(calls)
            let following = entries.dropFirst(index + 1).prefix(calls.count)
            let outputs: [Transcript.ToolOutput] = following.compactMap {
                if case .toolOutput(let output) = $0 { output } else { nil }
            }
            #expect(outputs.map(\.id) == calls.map(\.id), sourceLocation: sourceLocation)
            #expect(outputs.map(\.toolName) == calls.map(\.toolName), sourceLocation: sourceLocation)
        }
        #expect(!rounds.isEmpty, "The transcript has no tool calls", sourceLocation: sourceLocation)
        return rounds
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func toolHistoryOnNextTurn(backend: LlamaServerBackend) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: backend.model(at: baseURL), tools: [weatherTool])
        let first = try await session.respond(
            to: "Use the getWeather tool to find the weather in Paris. Reply with only: done"
        )
        attachResponse(first, "first")

        let second = try await session.respond(
            to: "Which city did you look up, and what was its temperature in °F?"
        )
        attachResponse(second, "second")

        #expect(await weatherTool.calls.count == 1)
        #expect(second.content.contains("Paris"))
        #expect(second.content.contains("72"))
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func toolCallWithStructuredOutput(backend: LlamaServerBackend) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: backend.model(at: baseURL), tools: [weatherTool])
        try await withKnownIssue(
            "With tools and a response schema, llama-server's grammar allows only the schema",
            isIntermittent: true
        ) {
            let response = try await session.respond(
                to: "Use the getWeather tool for Paris, then report the city and its temperature.",
                generating: WeatherReport.self
            )
            attachResponse(response)

            #expect(await weatherTool.calls.count == 1)
            #expect(response.content.city.contains("Paris"))
            #expect(response.content.temperatureFahrenheit == 72)
        } when: {
            !backend.callsToolsWithResponseSchema
        }
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func streamedToolCall(backend: LlamaServerBackend) async throws {
        let weatherTool = spy(on: WeatherTool())
        let session = LanguageModelSession(model: backend.model(at: baseURL), tools: [weatherTool])
        let response = try await session.streamResponse(
            to: "Use the getWeather tool to find the weather in Paris, then tell me the temperature."
        ).collect()
        attachResponse(response)

        #expect(await weatherTool.calls.count == 1)
        #expect(response.content.contains("72") || response.content.contains("23"))
    }

    // MARK: Reasoning

    @Test(arguments: LlamaServerBackend.allCases.filter(\.recordsReasoning))
    func reasoning(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.respond(to: "What is 17 × 3? Reply with only the number.")
        attachResponse(response)

        try #require(
            response.transcriptEntries.contains { if case .reasoning = $0 { true } else { false } },
            "The model gave no reasoning; reasoning needs a model that thinks."
        )
        #expect(response.content.contains("51"))
        #expect(!response.content.contains("<think>"))
    }

    @Test(arguments: LlamaServerBackend.allCases.filter(\.recordsReasoning))
    func streamedReasoning(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.streamResponse(
            to: "What is 17 × 3? Reply with only the number."
        ).collect()
        attachResponse(response)

        try #require(
            response.transcriptEntries.contains { if case .reasoning = $0 { true } else { false } },
            "The model gave no reasoning; reasoning needs a model that thinks."
        )
        #expect(response.content.contains("51"))
        #expect(!response.content.contains("<think>"))
    }

    // MARK: Generation options

    @Test(arguments: LlamaServerBackend.allCases)
    func temperatureZeroRepeats(backend: LlamaServerBackend) async throws {
        let model = backend.model(at: baseURL)
        let options = GenerationOptions(temperature: 0, maximumResponseTokens: 200)
        let prompt = "Write one sentence about the sea."
        let first = try await LanguageModelSession(model: model).respond(to: prompt, options: options)
        let second = try await LanguageModelSession(model: model).respond(to: prompt, options: options)
        attachResponse(first, "first")
        attachResponse(second, "second")

        #expect(first.content == second.content)
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func maximumResponseTokens(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.respond(
            to: "Write a long story about a lighthouse.",
            options: GenerationOptions(maximumResponseTokens: 8)
        )
        attachResponse(response)

        #expect(response.usage.output.totalTokenCount > 0)
        #expect(response.usage.output.totalTokenCount <= 8)
    }

    @Test(arguments: LlamaServerBackend.allCases.filter(\.hasStopSequences))
    func stopSequence(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.respond(
            to: "Count from 1 to 10, one number per line, with nothing else.",
            options: backend.options(stoppingAt: ["6"])
        )
        attachResponse(response)

        #expect(response.content.contains("5"))
        #expect(!response.content.contains("6"))
        #expect(!response.content.contains("9"))
    }

    // MARK: Cancellation

    @Test(arguments: LlamaServerBackend.allCases)
    func cancelStream(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let task = Task {
            var count = 0
            for try await _ in session.streamResponse(to: "Write a 2000-word story about a lighthouse.") {
                count += 1
                if count == 3 { break }
            }
            return count
        }
        #expect(try await task.value == 3)

        // The server stops generating when the client closes the connection.
        let deadline = ContinuousClock.now + .seconds(10)
        var idle = false
        while !idle, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(250))
            idle = try await serverIsIdle()
        }
        #expect(idle, "llama-server kept generating after the stream was cancelled")
    }

    private func serverIsIdle() async throws -> Bool {
        let (data, _) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("slots"))
        let slots = try JSONDecoder().decode([Slot].self, from: data)
        return slots.allSatisfy { !$0.isProcessing }
    }

    private struct Slot: Decodable {
        let isProcessing: Bool
        enum CodingKeys: String, CodingKey { case isProcessing = "is_processing" }
    }

    // MARK: Errors

    @Test(arguments: LlamaServerBackend.allCases)
    func promptLongerThanContext(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let prompt = String(repeating: "lighthouse ", count: 200_000)
        let error = try await #require(throws: HTTPTransportError.self) {
            try await session.respond(to: prompt)
        }
        #expect(error.response?.status.code == 400)
        let body = error.body.map { String(decoding: $0, as: UTF8.self) } ?? ""
        #expect(body.contains("exceed_context_size_error"))
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func streamedPromptLongerThanContext(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let prompt = String(repeating: "lighthouse ", count: 200_000)
        await #expect(throws: (any Error).self) {
            for try await _ in session.streamResponse(to: prompt) {}
        }
    }
}

@Suite(
    "llama-server vision",
    .serialized,
    .enabled(
        if: llamaServerVisionBaseURL != nil,
        "Set LLAMA_SERVER_VISION_BASE_URL to a llama-server with a vision model"
    )
)
struct LlamaServerVisionIntegrationTests {
    private let baseURL = llamaServerVisionBaseURL!

    @Test(arguments: LlamaServerBackend.allCases)
    func imageData(backend: LlamaServerBackend) async throws {
        let session = LanguageModelSession(model: backend.model(at: baseURL))
        let response = try await session.respond(
            to: "What color is this image? Reply with one word.",
            image: .init(data: redImageData, mimeType: "image/png")
        )
        attachResponse(response)

        #expect(response.content.localizedCaseInsensitiveContains("red"))
    }

    @Test(arguments: LlamaServerBackend.allCases)
    func imageInTranscript(backend: LlamaServerBackend) async throws {
        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(segments: [
                    .text(.init(content: "Remember this image.")),
                    .image(.init(data: redImageData, mimeType: "image/png")),
                ])
            ),
            .response(Transcript.Response(assetIDs: [], segments: [.text(.init(content: "OK."))])),
        ])
        let session = LanguageModelSession(model: backend.model(at: baseURL), transcript: transcript)
        let response = try await session.respond(to: "What color was the image? Reply with one word.")
        attachResponse(response)

        #expect(response.content.localizedCaseInsensitiveContains("red"))
    }
}
