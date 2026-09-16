@testable import AnyLanguageModel

struct MockLanguageModel: LanguageModel {
    enum UnavailableReason: Hashable, Sendable {
        case custom(String)
    }

    struct Request: Sendable {
        let schema: GenerationSchema
        let includeSchemaInPrompt: Bool
        let options: GenerationOptions
    }

    let requests = Locked<[Request]>([])

    var usage: LanguageModelSession.Usage = .zero
    var availabilityProvider: @Sendable () -> Availability<UnavailableReason>
    var responseProvider: @Sendable (Prompt, GenerationOptions) async throws -> String

    init(
        _ responseProvider:
            @escaping @Sendable (Prompt, GenerationOptions) async throws ->
            String = { _, _ in "Mock response" }
    ) {
        self.availabilityProvider = { .available }
        self.responseProvider = responseProvider
    }

    var availability: Availability<UnavailableReason> {
        return availabilityProvider()
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        try await respond(
            within: session,
            to: prompt,
            generating: type,
            schema: type.generationSchema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    func respond(
        within session: LanguageModelSession,
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<GeneratedContent> {
        try await respond(
            within: session,
            to: prompt,
            generating: GeneratedContent.self,
            schema: schema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    private func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        requests.withLock {
            $0.append(Request(schema: schema, includeSchemaInPrompt: includeSchemaInPrompt, options: options))
        }

        let promptWithInstructions = Prompt("Instructions: \(session.instructions?.description ?? "N/A")\n\(prompt)")
        let text = try await responseProvider(promptWithInstructions, options)

        let rawContent = try type == String.self ? GeneratedContent(text) : GeneratedContent(json: text)
        return LanguageModelSession.Response(
            content: try Content(rawContent),
            rawContent: rawContent,
            transcriptEntries: [],
            usage: usage
        )
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        streamResponse(
            within: session,
            to: prompt,
            generating: type,
            schema: type.generationSchema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    func streamResponse(
        within session: LanguageModelSession,
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<GeneratedContent> {
        streamResponse(
            within: session,
            to: prompt,
            generating: GeneratedContent.self,
            schema: schema,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    private func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        requests.withLock {
            $0.append(Request(schema: schema, includeSchemaInPrompt: includeSchemaInPrompt, options: options))
        }

        let promptWithInstructions = Prompt("Instructions: \(session.instructions?.description ?? "N/A")\n\(prompt)")

        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            Task {
                do {
                    let text = try await responseProvider(promptWithInstructions, options)
                    let generatedContent =
                        try type == String.self ? GeneratedContent(text) : GeneratedContent(json: text)
                    let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                        content: try Content(generatedContent).asPartiallyGenerated(),
                        rawContent: generatedContent,
                        usage: usage
                    )
                    continuation.yield(snapshot)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        return LanguageModelSession.ResponseStream(stream: stream)
    }
}

// MARK: -

extension MockLanguageModel {
    static var echo: Self {
        MockLanguageModel { prompt, _ in
            prompt.description
        }
    }

    static func fixed(_ response: String) -> Self {
        MockLanguageModel { _, _ in response }
    }

    static var unavailable: Self {
        var model = MockLanguageModel.echo
        model.availabilityProvider = { .unavailable(.custom("MockLanguageModel is unavailable")) }
        return model
    }

    static func streamingMock() -> Self {
        MockLanguageModel { _, _ in
            try await Task.sleep(for: .milliseconds(100))
            return "Streaming response"
        }
    }
}
