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
    /// The response that ends when a ``ToolExecutionDelegate`` stops a tool call
    /// before the model writes an answer.
    ///
    /// A `String`, or a type that can be empty, ends with an empty value.
    /// A type with required properties can't be empty,
    /// so this throws ``LanguageModelSession/ToolExecutionStopped`` with the entries instead.
    func stoppedResponse<Content: Generable>(
        generating type: Content.Type,
        provider: String,
        transcriptEntries: [Transcript.Entry],
        requests: RequestsUsage
    ) throws -> LanguageModelSession.Response<Content> {
        func response(_ content: Content, _ rawContent: GeneratedContent) -> LanguageModelSession.Response<Content> {
            .init(
                content: content,
                rawContent: rawContent,
                transcriptEntries: ArraySlice(transcriptEntries),
                requests: requests
            )
        }
        if type == String.self {
            return response("" as! Content, GeneratedContent(""))
        }
        let emptyObject = GeneratedContent(properties: [:])
        if let content = try? Content(emptyObject) {
            return response(content, emptyObject)
        }
        let nullContent = GeneratedContent(kind: .null)
        if let content = try? Content(nullContent) {
            return response(content, nullContent)
        }
        let emptyArray = GeneratedContent(kind: .array([]))
        if let content = try? Content(emptyArray) {
            return response(content, emptyArray)
        }
        throw LanguageModelSession.ToolExecutionStopped(
            provider: provider,
            transcriptEntries: ArraySlice(transcriptEntries)
        )
    }

    /// Constrained local generation produces one complete structured snapshot.
    func streamStructuredResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> LanguageModelSession.ResponseStream<Content> {
        streamStructuredResponse(within: session) {
            try await respond(
                within: session,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }
    }

    /// Converts a complete response into one snapshot without changing its schema.
    ///
    /// A stream counts its requests from its snapshots.
    /// A response that throws yields none,
    /// so the requests of the rounds that it completed are counted for `session` instead.
    /// A response that returns after the consumer cancels
    /// yields a snapshot that `session` doesn't read,
    /// so in a stream that `session` makes,
    /// its requests are prepaid before the snapshot is yielded,
    /// through the account where `session` counts the stream's snapshots,
    /// and the usage of the snapshot's final request carries the token for them.
    /// Within a response that isn't streamed, such as one that collects this stream,
    /// each round is counted through that response's channel as the round completes.
    func streamStructuredResponse<Content: Generable>(
        within session: LanguageModelSession,
        generate: @escaping @Sendable () async throws -> LanguageModelSession.Response<Content>
    ) -> LanguageModelSession.ResponseStream<Content> {
        .init(
            stream: AsyncThrowingStream { continuation in
                let enclosing = RequestUsageChannel.current
                let counted = enclosing?.streamedRequests
                let task = Task {
                    let channel =
                        enclosing.flatMap { $0.streamedRequests == nil ? $0 : nil }
                        ?? RequestUsageChannel { _ in }
                    do {
                        let response = try await RequestUsageChannel.$current.withValue(channel) {
                            try await generate()
                        }
                        var requests = response.requests
                        requests.final.prepaid = counted?.prepay(response.requests)
                        continuation.yield(
                            .init(
                                content: response.content.asPartiallyGenerated(),
                                rawContent: response.rawContent,
                                transcriptEntries: response.transcriptEntries,
                                requests: requests
                            )
                        )
                        continuation.finish()
                    } catch {
                        // No snapshot reports these requests,
                        // so they're added to what the session counted,
                        // whatever the stream's earlier snapshots reported.
                        if channel !== enclosing {
                            session.recordUsage(channel.reportedUsage)
                        }
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        )
    }
}
