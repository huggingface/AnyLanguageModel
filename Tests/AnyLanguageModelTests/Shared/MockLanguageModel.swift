@testable import AnyLanguageModel

struct MockLanguageModel: LanguageModel {
    enum UnavailableReason: Hashable, Sendable {
        case custom(String)
    }

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
        // For now, only String is supported
        guard type == String.self else {
            fatalError("MockLanguageModel only supports generating String content")
        }

        let promptWithInstructions = Prompt("Instructions: \(session.instructions?.description ?? "N/A")\n\(prompt)")
        let text = try await responseProvider(promptWithInstructions, options)

        return LanguageModelSession.Response(
            content: text as! Content,
            rawContent: GeneratedContent(text),
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
        // For now, only String is supported
        guard type == String.self else {
            fatalError("MockLanguageModel only supports generating String content")
        }

        let promptWithInstructions = Prompt("Instructions: \(session.instructions?.description ?? "N/A")\n\(prompt)")

        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            Task {
                do {
                    let text = try await responseProvider(promptWithInstructions, options)
                    let generatedContent = GeneratedContent(text)
                    let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                        content: (text as! Content).asPartiallyGenerated(),
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
