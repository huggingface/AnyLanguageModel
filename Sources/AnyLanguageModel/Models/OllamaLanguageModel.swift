import Foundation
import JSONSchema
import OrderedCollections

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A language model that connects to Ollama.
///
/// Use this model to generate text using models running locally with Ollama.
///
/// ```swift
/// let model = OllamaLanguageModel(model: "qwen2.5")
/// ```
public struct OllamaLanguageModel: LanguageModel {
    /// The reason the model is unavailable.
    /// This model is always available.
    public typealias UnavailableReason = Never

    /// Custom generation options specific to Ollama.
    ///
    /// Use this type to pass additional model parameters that are not part
    /// of the standard ``GenerationOptions``.
    ///
    /// Available options are model-specific and defined in the model's Modelfile.
    /// Common options include `seed`, `repeat_penalty`, `stop`, and others.
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: OllamaLanguageModel.self] = [
    ///     "seed": 42,
    ///     "repeat_penalty": 1.2
    /// ]
    /// ```
    ///
    /// - SeeAlso: [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
    public typealias CustomGenerationOptions = [String: JSONValue]

    /// The default base URL for Ollama.
    public static let defaultBaseURL = URL(string: "http://localhost:11434")!

    /// The base URL for the Ollama server.
    public let baseURL: URL

    /// The model identifier to use for generation.
    public let model: String

    private let urlSession: URLSession

    /// Creates an Ollama language model.
    ///
    /// - Parameters:
    ///   - baseURL: The base URL for the Ollama server. Defaults to `http://localhost:11434`.
    ///   - model: The model identifier (for example, "qwen2.5" or "llama3.3").
    ///   - session: The URL session to use for network requests.
    public init(
        baseURL: URL = defaultBaseURL,
        model: String,
        session: URLSession = URLSession(configuration: .default)
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.model = model
        self.urlSession = session
    }

    public func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
        let (ollamaText, ollamaImages) = convertSegmentsToOllama(userSegments)
        let messages = [
            OllamaMessage(role: .user, content: ollamaText)
        ]
        let ollamaOptions = convertOptions(options)
        let ollamaTools = try session.tools.map { tool in
            try convertToolToOllamaFormat(tool)
        }
        let ollamaFormat: JSONValue?
        if type == String.self {
            ollamaFormat = nil
        } else {
            let schema = try convertSchemaToOllamaFormat(type.generationSchema)
            ollamaFormat = try JSONValue(schema)
        }

        let params = try createChatParams(
            model: model,
            messages: messages,
            tools: ollamaTools.isEmpty ? nil : ollamaTools,
            options: ollamaOptions,
            stream: false,
            images: ollamaImages.isEmpty ? nil : ollamaImages,
            format: ollamaFormat
        )

        let url = baseURL.appendingPathComponent("api/chat")
        let body = try JSONEncoder().encode(params)
        let chatResponse: ChatResponse = try await urlSession.fetch(
            .post,
            url: url,
            body: body,
            dateDecodingStrategy: .iso8601WithFractionalSeconds
        )

        var entries: [Transcript.Entry] = []

        if let toolCalls = chatResponse.message.toolCalls, !toolCalls.isEmpty {
            let resolution = try await resolveToolCalls(toolCalls, session: session)
            switch resolution {
            case .stop(let calls):
                if !calls.isEmpty {
                    entries.append(.toolCalls(Transcript.ToolCalls(calls)))
                }
                return LanguageModelSession.Response(
                    content: "" as! Content,
                    rawContent: GeneratedContent(""),
                    transcriptEntries: ArraySlice(entries)
                )
            case .invocations(let invocations):
                if !invocations.isEmpty {
                    entries.append(.toolCalls(Transcript.ToolCalls(invocations.map(\.call))))
                    for invocation in invocations {
                        entries.append(.toolOutput(invocation.output))
                    }
                }
            }
        }

        let text = chatResponse.message.content ?? ""
        if type == String.self {
            return LanguageModelSession.Response(
                content: text as! Content,
                rawContent: GeneratedContent(text),
                transcriptEntries: ArraySlice(entries)
            )
        }

        let generatedContent = try GeneratedContent(json: text)
        let content = try type.init(generatedContent)
        return LanguageModelSession.Response(
            content: content,
            rawContent: generatedContent,
            transcriptEntries: ArraySlice(entries)
        )
    }

    public func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let userSegments = extractPromptSegments(from: session, fallbackText: prompt.description)
        let (ollamaText, ollamaImages) = convertSegmentsToOllama(userSegments)
        let messages = [
            OllamaMessage(role: .user, content: ollamaText)
        ]
        let ollamaOptions = convertOptions(options)
        let url = baseURL.appendingPathComponent("api/chat")

        // Transform the newline-delimited JSON stream from Ollama into ResponseStream snapshots
        let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
            AsyncThrowingStream { continuation in
                do {
                    let ollamaTools = try session.tools.map { tool in
                        try convertToolToOllamaFormat(tool)
                    }
                    let ollamaFormat: JSONValue?
                    if type == String.self {
                        ollamaFormat = nil
                    } else {
                        let schema = try convertSchemaToOllamaFormat(type.generationSchema)
                        ollamaFormat = try JSONValue(schema)
                    }

                    let params = try createChatParams(
                        model: model,
                        messages: messages,
                        tools: ollamaTools.isEmpty ? nil : ollamaTools,
                        options: ollamaOptions,
                        stream: true,
                        images: (ollamaImages.isEmpty ? nil : ollamaImages),
                        format: ollamaFormat
                    )
                    let body = try JSONEncoder().encode(params)

                    let task = Task {
                        // Reuse ChatResponse as each streamed line shares the same shape
                        do {
                            let chunks =
                                urlSession.fetchStream(
                                    .post,
                                    url: url,
                                    body: body,
                                    dateDecodingStrategy: .iso8601WithFractionalSeconds
                                ) as AsyncThrowingStream<ChatResponse, any Error>

                            var partialText = ""

                            for try await chunk in chunks {
                                if let piece = chunk.message.content {
                                    partialText += piece
                                    if type == String.self {
                                        let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                                            content: (partialText as! Content).asPartiallyGenerated(),
                                            rawContent: GeneratedContent(partialText)
                                        )
                                        continuation.yield(snapshot)
                                    } else if let raw = try? GeneratedContent(json: partialText),
                                        let parsed = try? type.init(raw)
                                    {
                                        let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                                            content: parsed.asPartiallyGenerated(),
                                            rawContent: raw
                                        )
                                        continuation.yield(snapshot)
                                    } else {
                                        // Structured responses can stream as incomplete JSON fragments.
                                        // Skip snapshots until the accumulated JSON parses cleanly.
                                    }
                                }

                                if chunk.done {
                                    break
                                }
                            }

                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }

                    continuation.onTermination = { _ in
                        task.cancel()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }

        return LanguageModelSession.ResponseStream(stream: stream)
    }
}

// MARK: - Tool Invocation Handling

private struct ToolInvocationResult {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

private enum ToolResolutionOutcome {
    case stop(calls: [Transcript.ToolCall])
    case invocations([ToolInvocationResult])
}

private func resolveToolCalls(
    _ toolCalls: [OllamaToolCall],
    session: LanguageModelSession
) async throws -> ToolResolutionOutcome {
    if toolCalls.isEmpty {
        return .invocations([])
    }

    var toolsByName: [String: any Tool] = [:]
    for tool in session.tools {
        if toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }
    }

    var transcriptCalls: [Transcript.ToolCall] = []
    transcriptCalls.reserveCapacity(toolCalls.count)
    for call in toolCalls {
        let args = try toGeneratedContent(call.function.arguments)
        let callID = call.id ?? UUID().uuidString
        transcriptCalls.append(
            Transcript.ToolCall(
                id: callID,
                toolName: call.function.name,
                arguments: args
            )
        )
    }

    if let delegate = session.toolExecutionDelegate {
        await delegate.didGenerateToolCalls(transcriptCalls, in: session)
    }

    guard !transcriptCalls.isEmpty else { return .invocations([]) }

    var decisions: [ToolExecutionDecision] = []
    decisions.reserveCapacity(transcriptCalls.count)

    if let delegate = session.toolExecutionDelegate {
        for call in transcriptCalls {
            let decision = await delegate.toolCallDecision(for: call, in: session)
            if case .stop = decision {
                return .stop(calls: transcriptCalls)
            }
            decisions.append(decision)
        }
    } else {
        decisions = Array(repeating: .execute, count: transcriptCalls.count)
    }

    var results: [ToolInvocationResult] = []
    results.reserveCapacity(transcriptCalls.count)

    for (index, call) in transcriptCalls.enumerated() {
        switch decisions[index] {
        case .stop:
            // This branch should be unreachable because `.stop` returns during decision collection.
            // Keep it as a defensive guard in case that logic changes.
            return .stop(calls: transcriptCalls)
        case .provideOutput(let segments):
            let output = Transcript.ToolOutput(
                id: call.id,
                toolName: call.toolName,
                segments: segments
            )
            if let delegate = session.toolExecutionDelegate {
                await delegate.didExecuteToolCall(call, output: output, in: session)
            }
            results.append(ToolInvocationResult(call: call, output: output))
        case .execute:
            guard let tool = toolsByName[call.toolName] else {
                let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.toolName)"))
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: [message]
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(ToolInvocationResult(call: call, output: output))
                continue
            }

            do {
                let segments = try await tool.makeOutputSegments(from: call.arguments)
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: tool.name,
                    segments: segments
                )
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                results.append(ToolInvocationResult(call: call, output: output))
            } catch {
                if let delegate = session.toolExecutionDelegate {
                    await delegate.didFailToolCall(call, error: error, in: session)
                }
                throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
            }
        }
    }

    return .invocations(results)
}

// MARK: - Conversions

private func convertOptions(_ options: GenerationOptions) -> [String: JSONValue]? {
    var ollamaOptions: [String: JSONValue] = [:]

    // Handle temperature
    if let temperature = options.temperature {
        ollamaOptions["temperature"] = .double(temperature)
    }

    // Greedy sampling uses temperature = 0 for deterministic output
    if case .greedy? = options.sampling?.mode {
        ollamaOptions["temperature"] = .double(0.0)
    }

    // Handle maximum response tokens
    if let maxTokens = options.maximumResponseTokens {
        ollamaOptions["num_predict"] = .int(maxTokens)
    }

    // Handle sampling mode specific parameters
    if let sampling = options.sampling {
        switch sampling.mode {
        case .greedy:
            break

        case .topK(let k, let seed):
            ollamaOptions["top_k"] = .int(k)
            if let seed = seed {
                ollamaOptions["seed"] = .int(Int(seed))
            }

        case .nucleus(let probabilityThreshold, let seed):
            ollamaOptions["top_p"] = .double(probabilityThreshold)
            if let seed = seed {
                ollamaOptions["seed"] = .int(Int(seed))
            }
        }
    }

    // Merge custom Ollama options
    if let customOptions: [String: JSONValue] = options[custom: OllamaLanguageModel.self] {
        for (key, value) in customOptions {
            ollamaOptions[key] = value
        }
    }

    return ollamaOptions.isEmpty ? nil : ollamaOptions
}

private func convertToolToOllamaFormat(_ tool: any Tool) throws -> [String: JSONValue] {
    let resolvedSchema = tool.parameters.withResolvedRoot() ?? tool.parameters
    return [
        "type": .string("function"),
        "function": .object([
            "name": .string(tool.name),
            "description": .string(tool.description),
            "parameters": try JSONValue(resolvedSchema),
        ]),
    ]
}

private func convertSchemaToOllamaFormat(_ schema: GenerationSchema) throws -> JSONSchema {
    let resolvedSchema = schema.withResolvedRoot() ?? schema
    let data = try JSONEncoder().encode(resolvedSchema)
    return try JSONDecoder().decode(JSONSchema.self, from: data)
}

private func toGeneratedContent(_ value: JSONValue?) throws -> GeneratedContent {
    guard let value else { return GeneratedContent(properties: [:]) }
    let data = try JSONEncoder().encode(value)
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}

private func createChatParams(
    model: String,
    messages: [OllamaMessage],
    tools: [[String: JSONValue]]?,
    options: [String: JSONValue]?,
    stream: Bool,
    images: [String]?,
    format: JSONValue?
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "model": .string(model),
        "messages": try JSONValue(messages),
        "stream": .bool(stream),
    ]

    if let tools {
        params["tools"] = try JSONValue(tools)
    }

    if let options {
        params["options"] = .object(options)
    }

    if let images, !images.isEmpty {
        params["images"] = .array(images.map { .string($0) })
    }

    if let format {
        params["format"] = format
    }

    return params
}

// MARK: - Supporting Types

private struct OllamaMessage: Hashable, Codable, Sendable {
    enum Role: String, Hashable, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    let role: Role
    let content: String
}

private func convertSegmentsToOllama(_ segments: [Transcript.Segment]) -> (String, [String]) {
    var textParts: [String] = []
    var images: [String] = []
    for segment in segments {
        switch segment {
        case .text(let t):
            textParts.append(t.content)
        case .structure(let s):
            textParts.append(s.content.jsonString)
        case .image(let img):
            switch img.source {
            case .data(let data, _):
                images.append(data.base64EncodedString())
            case .url(let url):
                // Ollama supports base64 images; include URL as text if provided
                textParts.append(url.absoluteString)
            }
        }
    }
    return (textParts.joined(separator: "\n"), images)
}

private func extractPromptSegments(from session: LanguageModelSession, fallbackText: String) -> [Transcript.Segment] {
    for entry in session.transcript.reversed() {
        if case .prompt(let p) = entry {
            return p.segments
        }
    }
    return [.text(.init(content: fallbackText))]
}

private struct ChatResponse: Decodable, Sendable {
    let model: String
    let createdAt: Date
    let message: ChatMessageResponse
    let done: Bool

    private enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case message
        case done
    }
}

private struct ChatMessageResponse: Decodable, Sendable {
    let role: OllamaMessage.Role
    let content: String?
    let toolCalls: [OllamaToolCall]?

    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
    }
}

private struct OllamaToolCall: Decodable, Sendable {
    let id: String?
    let type: String?
    let function: OllamaToolFunction
}

private struct OllamaToolFunction: Decodable, Sendable {
    let name: String
    let arguments: JSONValue?

    private enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }
}
