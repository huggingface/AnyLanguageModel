import Testing

@testable import AnyLanguageModel

@Suite("Local generation token usage")
struct LocalGenerationUsageTests {
    @Test(arguments: [0, 4])
    func completePromptIncludesReusedPrefix(cachedTokens: Int) {
        let usage = LocalGenerationUsage(
            promptTokenCount: 10,
            cachedTokenCount: cachedTokens,
            generatedTokenCount: 3
        ).value
        #expect(usage.input.totalTokenCount == 10)
        #expect(usage.input.cachedTokenCount == cachedTokens)
        #expect(usage.output.totalTokenCount == 3)
        #expect(usage.totalTokenCount == 13)
        #expect(usage.output.reasoningTokenCount == 0)
        #expect(usage.metadata.isEmpty)
    }

    @Test func acceptedTokensCountBeforeDecodingAndFiltering() {
        var counts = LocalGenerationUsage(promptTokenCount: 7)
        // One invisible token, a tool marker that is filtered, visible text, then EOG.
        let decoded: [String?] = [nil, "<tool_call>", "hello", "<eog>"]
        var visible = ""
        for index in decoded.indices {
            guard counts.acceptToken(isEndOfGeneration: index == 3) else { break }
            if let text = decoded[index], !text.hasPrefix("<") { visible += text }
        }
        #expect(visible == "hello")
        #expect(counts.value.output.totalTokenCount == 3)
    }

    @Test func emptyGenerationExcludesEndToken() {
        var counts = LocalGenerationUsage(promptTokenCount: 7)
        let accepted = counts.acceptToken(isEndOfGeneration: true)
        #expect(!accepted)
        #expect(counts.value.input.totalTokenCount == 7)
        #expect(counts.value.output.totalTokenCount == 0)
        #expect(counts.value.input.cachedTokenCount == 0)
    }

    @Test func cumulativeCallbacksAndFinalSequenceReconcileInvisibleTokens() throws {
        var stream = LocalGenerationTokenStream(promptTokenCount: 2)
        let decode: ([Int]) -> String = { $0.contains(3) ? "hello" : "" }
        let firstUpdate = stream.update([1, 2, 3], decode: decode)
        let first = try #require(firstUpdate)
        #expect(first.text == "hello")
        #expect(first.usage.output.totalTokenCount == 1)
        let invisibleUpdate = stream.update([1, 2, 3, 4], decode: decode)
        let invisible = try #require(invisibleUpdate)
        #expect(invisible.text == first.text)
        #expect(invisible.usage.output.totalTokenCount == 2)
        let repeated = stream.update([1, 2, 3, 4], decode: decode)
        #expect(repeated == nil)
        // A final sequence can contain tokens that were not included in callbacks.
        let finalUpdate = stream.update([1, 2, 3, 4, 5], decode: decode)
        let final = try #require(finalUpdate)
        #expect(final.text == first.text)
        #expect(final.usage.input.totalTokenCount == 2)
        #expect(final.usage.output.totalTokenCount == 3)
        #expect(final.usage.input.cachedTokenCount == 0)
    }

    @Test func finalSequenceWithoutCallbacksPublishesPromptUsage() throws {
        var stream = LocalGenerationTokenStream(promptTokenCount: 2)
        let finalUpdate = stream.update([1, 2], decode: { _ in "" })
        let update = try #require(finalUpdate)
        #expect(update.text.isEmpty)
        #expect(update.usage.input.totalTokenCount == 2)
        #expect(update.usage.output.totalTokenCount == 0)
        let repeated = stream.update([1, 2], decode: { _ in "" })
        #expect(repeated == nil)
    }

    @Test func shortReturnedSequencePreservesExistingFallback() {
        let counts = LocalGenerationUsage(promptTokenCount: 3)
        #expect(Array(counts.generatedTokens(in: [4, 5])) == [4, 5])
        #expect(counts.generatedTokens(in: []).isEmpty)
    }

    #if Llama
        @Test func multimodalCountsUseChunkTokensInsteadOfPositions() {
            let chunks = [(tokens: 8, positions: 8), (tokens: 256, positions: 16), (tokens: 3, positions: 3)]
            var visited: [Int] = []
            let total = LlamaLanguageModel.multimodalPromptTokenCount(chunkCount: chunks.count) { index in
                visited.append(index)
                return chunks[index].tokens
            }
            #expect(visited == [0, 1, 2])
            #expect(total == 267)
            #expect(total != chunks.reduce(0) { $0 + $1.positions })
            #expect(LlamaLanguageModel.multimodalPromptTokenCount(chunkCount: 0) { _ in 1 } == 0)
        }
    #endif

    #if MLX
        @Test func cacheReuseRequiresMatchingPreparedPrefixAndConfiguration() {
            func cachedCount(
                prefix: [Int32] = [1, 2],
                prefill: Int = 2,
                current: [Int32] = [1, 2, 3],
                configurationMatches: Bool = true,
                hasMedia: Bool = false
            ) -> Int {
                MLXLanguageModel.reusablePrefixTokenCount(
                    prefixTokens: prefix,
                    prefillTokenCount: prefill,
                    currentTokens: current,
                    configurationMatches: configurationMatches,
                    hasMedia: hasMedia
                )
            }
            #expect(cachedCount() == 2)
            #expect(cachedCount(current: [1, 9, 3]) == 0)
            #expect(cachedCount(current: [1, 2]) == 0)
            #expect(cachedCount(current: [1]) == 0)
            #expect(cachedCount(prefill: 1) == 0)
            #expect(cachedCount(prefix: [], prefill: 0) == 0)
            #expect(cachedCount(configurationMatches: false) == 0)
            #expect(cachedCount(hasMedia: true) == 0)
        }
    #endif

    @Test func structuredSnapshotPreservesUsageAndSessionAccumulation() async throws {
        let model = LocalStructuredUsageModel()
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: "A number", generating: Int.self)
        let streamed = try await session.streamResponse(to: "A number", generating: Int.self).collect()
        #expect(response.content == 7)
        #expect(streamed.content == response.content)
        #expect(streamed.usage == response.usage)
        #expect(streamed.usage == model.usage)
        #expect(session.usage.totalTokenCount == 2 * model.usage.totalTokenCount)
    }
}

private struct LocalStructuredUsageModel: LanguageModel {
    typealias UnavailableReason = Never
    let usage = LocalGenerationUsage(promptTokenCount: 12, generatedTokenCount: 1).value

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        let raw = GeneratedContent(7)
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [], usage: usage)
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        streamStructuredResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }
}
