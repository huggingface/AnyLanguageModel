/// Request-local counts from native token generation, independent of text decoding.
struct LocalGenerationUsage {
    let promptTokenCount: Int
    var cachedTokenCount: Int = 0
    var generatedTokenCount: Int = 0

    var value: LanguageModelSession.Usage {
        .init(
            input: .init(totalTokenCount: promptTokenCount, cachedTokenCount: cachedTokenCount),
            output: .init(totalTokenCount: generatedTokenCount, reasoningTokenCount: 0)
        )
    }

    /// Count accepted output before decoding or filtering it, excluding discarded end tokens.
    mutating func acceptToken(isEndOfGeneration: Bool) -> Bool {
        guard !isEndOfGeneration else { return false }
        generatedTokenCount += 1
        return true
    }

    /// swift-transformers returns the prompt followed by the generated suffix.
    /// Preserve the adapter's fallback for sequences that do not echo the full prompt.
    func generatedTokens(in tokenIDs: [Int]) -> ArraySlice<Int> {
        tokenIDs.count >= promptTokenCount
            ? tokenIDs.dropFirst(promptTokenCount)
            : tokenIDs[tokenIDs.indices]
    }
}

/// Reconciles cumulative native callbacks and the final prompt-prefixed sequence.
struct LocalGenerationTokenStream {
    private var counts: LocalGenerationUsage
    private var lastText: String?
    private var lastUsage: LanguageModelSession.Usage?

    init(promptTokenCount: Int) {
        counts = LocalGenerationUsage(promptTokenCount: promptTokenCount)
    }

    mutating func update(
        _ tokenIDs: [Int],
        decode: ([Int]) -> String
    ) -> (text: String, usage: LanguageModelSession.Usage)? {
        let generatedTokens = counts.generatedTokens(in: tokenIDs)
        counts.generatedTokenCount = generatedTokens.count
        let text = decode(Array(generatedTokens))
        let usage = counts.value
        guard text != lastText || usage != lastUsage else { return nil }
        lastText = text
        lastUsage = usage
        return (text, usage)
    }
}

extension LanguageModel {
    /// Constrained local generation produces one complete structured snapshot.
    func streamStructuredResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> LanguageModelSession.ResponseStream<Content> {
        .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        let response = try await respond(
                            within: session,
                            to: prompt,
                            generating: type,
                            includeSchemaInPrompt: includeSchemaInPrompt,
                            options: options
                        )
                        continuation.yield(
                            .init(
                                content: response.content.asPartiallyGenerated(),
                                rawContent: response.rawContent,
                                transcriptEntries: response.transcriptEntries,
                                usage: response.usage
                            )
                        )
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        )
    }
}
