import Testing

@testable import AnyLanguageModel

@Suite("Local generation token usage")
struct LocalGenerationUsageTests {
    /// A response's rounds, as MLX and llama.cpp count them: the final request is the latest one
    /// added, and the total adds every request up.
    @Test func requestsUsageKeepsTheFinalRequestAndTheTotal() {
        func usage(_ input: Int, _ output: Int) -> LanguageModelSession.Usage {
            .init(
                input: .init(totalTokenCount: input, cachedTokenCount: 0),
                output: .init(totalTokenCount: output, reasoningTokenCount: 0)
            )
        }
        var requests = RequestsUsage()
        requests.add(usage(10, 2))
        requests.add(usage(15, 3))
        #expect(requests.final == usage(15, 3))
        #expect(requests.total == usage(25, 5))
    }

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
                cacheOffset: Int = 2,
                current: [Int32] = [1, 2, 3],
                configurationMatches: Bool = true,
                hasMedia: Bool = false,
                canTrim: Bool = false
            ) -> Int {
                MLXLanguageModel.reusablePrefixTokenCount(
                    prefixTokens: prefix,
                    prefillTokenCount: prefill,
                    cacheOffset: cacheOffset,
                    currentTokens: current,
                    configurationMatches: configurationMatches,
                    hasMedia: hasMedia,
                    canTrim: canTrim
                )
            }
            #expect(cachedCount() == 2)
            #expect(cachedCount(current: [1, 9, 3]) == 0)
            #expect(cachedCount(current: [1, 2]) == 0)
            #expect(cachedCount(current: [1]) == 0)
            #expect(cachedCount(prefill: 1) == 0)
            #expect(cachedCount(cacheOffset: 5) == 0)

            // A trimmable cache can reuse the longest common prefix,
            // leaving at least one token to evaluate.
            #expect(cachedCount(current: [1, 9, 3], canTrim: true) == 1)
            #expect(cachedCount(current: [1, 2], canTrim: true) == 1)
            #expect(cachedCount(current: [1], canTrim: true) == 0)
            #expect(cachedCount(current: [9, 2, 3], canTrim: true) == 0)
            #expect(cachedCount(canTrim: true) == 2)
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

    /// A structured response that called tools made a request for each round,
    /// and a streamed one counts every round, as a response does.
    @Test func structuredSnapshotCountsEveryRequest() async throws {
        let model = LocalStructuredUsageModel(
            earlier: LocalGenerationUsage(promptTokenCount: 30, generatedTokenCount: 5).value
        )
        let streamedSession = LanguageModelSession(model: model)
        let streamed = try await streamedSession.streamResponse(to: "A number", generating: Int.self).collect()
        let respondedSession = LanguageModelSession(model: model)
        let response = try await respondedSession.respond(to: "A number", generating: Int.self)
        #expect(streamed.usage == model.usage)
        #expect(response.usage == model.usage)
        #expect(respondedSession.usage.totalTokenCount == 48)
        #expect(streamedSession.usage == respondedSession.usage)
    }

    /// A stopped response ends with an empty value when its type can be empty,
    /// and otherwise throws with the tool calls that the delegate stopped.
    @Test func stoppedResponseThrowsWithEntriesForRequiredFields() throws {
        let model = LocalStructuredUsageModel()
        let call = Transcript.ToolCall(
            id: "call-1",
            toolName: WeatherTool().name,
            arguments: try GeneratedContent(json: #"{"city":"Paris"}"#)
        )
        let entries: [Transcript.Entry] = [.toolCalls(.init([call]))]

        let text = try model.stoppedResponse(
            generating: String.self,
            provider: "Local",
            transcriptEntries: entries,
            requests: RequestsUsage()
        )
        #expect(text.content.isEmpty)
        #expect(Array(text.transcriptEntries) == entries)

        let reports = try model.stoppedResponse(
            generating: [WeatherReport].self,
            provider: "Local",
            transcriptEntries: entries,
            requests: RequestsUsage()
        )
        #expect(reports.content.isEmpty)
        #expect(reports.rawContent == GeneratedContent(kind: .array([])))
        #expect(Array(reports.transcriptEntries) == entries)

        #expect {
            _ = try model.stoppedResponse(
                generating: WeatherReport.self,
                provider: "Local",
                transcriptEntries: entries,
                requests: RequestsUsage()
            )
        } throws: { error in
            guard let stopped = error as? LanguageModelSession.ToolExecutionStopped else { return false }
            return stopped.provider == "Local" && Array(stopped.transcriptEntries) == entries
        }

        func expectStopped<Content: Generable>(_ type: Content.Type) {
            #expect(throws: LanguageModelSession.ToolExecutionStopped.self) {
                _ = try model.stoppedResponse(
                    generating: type,
                    provider: "Local",
                    transcriptEntries: entries,
                    requests: RequestsUsage()
                )
            }
        }
        expectStopped(Int.self)
        expectStopped(Bool.self)
    }

    /// A streamed structured response that throws after a completed round yields no snapshot,
    /// and the session still counts the round's request, as a response that isn't streamed does.
    @Test(arguments: [false, true])
    func failedStructuredStreamCountsCompletedRounds(_ streaming: Bool) async throws {
        let model = LocalFailingRoundModel()
        let session = LanguageModelSession(model: model)
        do {
            _ =
                try await streaming
                ? session.streamResponse(to: "A number", generating: Int.self).collect()
                : session.respond(to: "A number", generating: Int.self)
            Issue.record("Expected LocalFailingRoundModel.Failure")
        } catch is LocalFailingRoundModel.Failure {}
        #expect(session.usage == model.round)
    }

    /// A structured response that reports its rounds as they complete
    /// counts each request once, whether or not it's streamed.
    @Test(arguments: [false, true])
    func structuredRoundsReportedAsTheyCompleteCountOnce(_ streaming: Bool) async throws {
        let model = LocalReportingRoundsModel()
        let session = LanguageModelSession(model: model)
        let usage =
            try await streaming
            ? session.streamResponse(to: "A number", generating: Int.self).collect().usage
            : session.respond(to: "A number", generating: Int.self).usage
        #expect(usage == model.second)
        #expect(session.usage == model.total)
    }

    /// A consumer that cancels a structured stream while a round is still in progress
    /// stops reading before the response returns or throws,
    /// and the session still counts the round's request once.
    @Test(arguments: [false, true])
    func cancelledStructuredStreamCountsCompletedRounds(_ throwsCancellation: Bool) async throws {
        let model = LocalCancelledRoundModel(throwsCancellation: throwsCancellation)
        let session = LanguageModelSession(model: model)
        let consumer = Task {
            for try await _ in session.streamResponse(to: "Numbers", generating: [Int].self) {}
        }
        await model.reported.wait()
        consumer.cancel()
        _ = await consumer.result
        model.release.open()
        #expect(try await eventually(session.usage != .zero))
        #expect(session.usage == model.round)
    }

    /// A snapshot that the stream yields
    /// and that the consumer cancels before the session reads it
    /// still counts its requests once.
    @Test func structuredSnapshotCancelledBeforeTheSessionReadsItCountsOnce() async throws {
        let model = LocalRelayedSnapshotModel()
        let session = LanguageModelSession(model: model)
        let consumer = Task {
            for try await _ in session.streamResponse(to: "Numbers", generating: [Int].self) {}
        }
        await model.held.wait()
        consumer.cancel()
        _ = await consumer.result
        model.release.open()
        await model.relayed.wait()
        #expect(try await eventually(session.usage != .zero))
        #expect(session.usage == model.round)
    }

    /// A model that wraps a structured stream counts its request once,
    /// whether its response collects the stream
    /// or its stream rebuilds each snapshot with the public initializer.
    @Test(arguments: [false, true])
    func wrappedStructuredStreamCountsItsRequestOnce(_ streaming: Bool) async throws {
        let model = LocalWrappingModel()
        let session = LanguageModelSession(model: model)
        let response =
            try await streaming
            ? session.streamResponse(to: "A number", generating: Int.self).collect()
            : session.respond(to: "A number", generating: Int.self)
        #expect(response.content == 7)
        #expect(session.usage == model.inner.round)
    }

    /// A stream that reports an earlier request
    /// and then relays a structured stream that completes a round and throws
    /// counts both requests.
    @Test func relayedStructuredStreamThatThrowsCountsEveryRequest() async throws {
        let model = LocalPrefixedFailingModel()
        let session = LanguageModelSession(model: model)
        await #expect(throws: LocalFailingRoundModel.Failure.self) {
            try await session.streamResponse(to: "A number", generating: Int.self).collect()
        }
        var total = model.earlier
        total.add(model.child.round)
        #expect(session.usage == total)
    }

    /// A stream that reports an earlier request
    /// and then relays a structured stream that completes after the consumer cancels
    /// counts both requests.
    @Test func relayedStructuredStreamCompletedAfterCancellationCountsEveryRequest() async throws {
        let model = LocalPrefixedCancelledModel()
        let session = LanguageModelSession(model: model)
        let received = Gate()
        let consumer = Task {
            for try await _ in session.streamResponse(to: "Numbers", generating: [Int].self) {
                received.open()
            }
        }
        await received.wait()
        await model.child.reported.wait()
        consumer.cancel()
        _ = await consumer.result
        model.child.release.open()
        #expect(try await eventually(session.usage != model.earlier))
        var total = model.earlier
        total.add(model.child.round)
        #expect(session.usage == total)
    }

    /// A stream whose own request reports usage before and after it relays a structured stream,
    /// or after the structured stream completes,
    /// counts the structured stream's request once and every increase of its own request.
    @Test(arguments: [false, true])
    func interleavedStructuredStreamCountsEveryRequestOnce(childFirst: Bool) async throws {
        let model = LocalInterleavedModel(childFirst: childFirst)
        let session = LanguageModelSession(model: model)
        var received = 0
        for try await _ in session.streamResponse(to: "A number", generating: Int.self) {
            received += 1
            if received == 1 { model.relayChild.open() }
            if received == 2 { model.reportAgain.open() }
        }
        var total = childFirst ? model.first : model.second
        total.add(model.child.round)
        #expect(session.usage == total)
    }
}

/// Waits for `condition` to become true, and returns false if it doesn't before `timeout`.
private func eventually(
    timeout: Duration = .seconds(2),
    _ condition: @autoclosure () -> Bool
) async throws -> Bool {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        if ContinuousClock.now >= deadline { return false }
        try await Task.sleep(for: .milliseconds(5))
    }
    return true
}

/// A one-time signal that a task waits for whether or not it's cancelled,
/// as a delegate that answers after the consumer cancels does.
private final class Gate: Sendable {
    private struct State: Sendable {
        var isOpen = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state = Locked(State())

    func open() {
        let waiters = state.withLock { state in
            state.isOpen = true
            defer { state.waiters = [] }
            return state.waiters
        }
        for waiter in waiters { waiter.resume() }
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            let isOpen = state.withLock { state in
                if !state.isOpen { state.waiters.append(continuation) }
                return state.isOpen
            }
            if isOpen { continuation.resume() }
        }
    }
}

/// A model that reports each of its two rounds as it completes, as MLX and llama.cpp do.
private struct LocalReportingRoundsModel: LanguageModel {
    typealias UnavailableReason = Never
    let first = LocalGenerationUsage(promptTokenCount: 30, generatedTokenCount: 5).value
    let second = LocalGenerationUsage(promptTokenCount: 40, generatedTokenCount: 2).value
    var total: LanguageModelSession.Usage {
        var total = first
        total.add(second)
        return total
    }

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        var requests = RequestsUsage()
        requests.add(first)
        requests.add(second)
        let raw = GeneratedContent(7)
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [], requests: requests)
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

/// A model whose first round completes and that then waits for `release`,
/// as a tool loop waits for a delegate.
/// It then stops with an empty response,
/// or throws `CancellationError` when `throwsCancellation` is true.
private struct LocalCancelledRoundModel: LanguageModel {
    typealias UnavailableReason = Never
    let throwsCancellation: Bool
    let round = LocalGenerationUsage(promptTokenCount: 20, generatedTokenCount: 4).value
    let reported = Gate()
    let release = Gate()

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        var requests = RequestsUsage()
        requests.add(round)
        reported.open()
        await release.wait()
        if throwsCancellation { throw CancellationError() }
        return try stoppedResponse(
            generating: type,
            provider: "Local",
            transcriptEntries: [],
            requests: requests
        )
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

/// A model that relays the snapshot of a structured stream,
/// holding it until `release` opens, so that the session doesn't read it before the consumer cancels.
private struct LocalRelayedSnapshotModel: LanguageModel {
    typealias UnavailableReason = Never
    let round = LocalGenerationUsage(promptTokenCount: 20, generatedTokenCount: 4).value
    let held = Gate()
    let release = Gate()
    let relayed = Gate()

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        var requests = RequestsUsage()
        requests.add(round)
        return try stoppedResponse(
            generating: type,
            provider: "Local",
            transcriptEntries: [],
            requests: requests
        )
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let upstream = streamStructuredResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
        let (held, release, relayed) = (held, release, relayed)
        return .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    defer { relayed.open() }
                    do {
                        for try await snapshot in upstream {
                            held.open()
                            await release.wait()
                            continuation.yield(snapshot)
                        }
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

/// A model whose first round completes
/// and whose response then throws, as when a tool throws.
private struct LocalFailingRoundModel: LanguageModel {
    typealias UnavailableReason = Never
    struct Failure: Error {}
    let round = LocalGenerationUsage(promptTokenCount: 20, generatedTokenCount: 4).value

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        var requests = RequestsUsage()
        requests.add(round)
        throw Failure()
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

private struct LocalStructuredUsageModel: LanguageModel {
    typealias UnavailableReason = Never
    var earlier = LanguageModelSession.Usage.zero
    let usage = LocalGenerationUsage(promptTokenCount: 12, generatedTokenCount: 1).value

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        let raw = GeneratedContent(7)
        return .init(
            content: try Content(raw),
            rawContent: raw,
            transcriptEntries: [],
            requests: RequestsUsage(earlier: earlier, final: usage)
        )
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

/// A model whose structured response makes one request.
private struct LocalSingleRoundModel: LanguageModel {
    typealias UnavailableReason = Never
    let round = LocalGenerationUsage(promptTokenCount: 20, generatedTokenCount: 4).value

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        var requests = RequestsUsage()
        requests.add(round)
        let raw = GeneratedContent(7)
        return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [], requests: requests)
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

/// A model of another library that wraps a local model:
/// its response collects the local model's stream,
/// and its stream rebuilds each snapshot with the public initializer.
private struct LocalWrappingModel: LanguageModel {
    typealias UnavailableReason = Never
    let inner = LocalSingleRoundModel()

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        try await inner.streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        ).collect()
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let upstream = inner.streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
        return .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        for try await snapshot in upstream {
                            continuation.yield(
                                .init(content: snapshot.content, rawContent: snapshot.rawContent, usage: snapshot.usage)
                            )
                        }
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

/// A model whose stream yields a snapshot with the usage of an earlier request,
/// then relays the stream of a local model whose round completes and whose response throws.
private struct LocalPrefixedFailingModel: LanguageModel {
    typealias UnavailableReason = Never
    let earlier = LocalGenerationUsage(promptTokenCount: 30, generatedTokenCount: 5).value
    let child = LocalFailingRoundModel()

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        try await streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        ).collect()
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let (earlier, child) = (earlier, child)
        return .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        let raw = GeneratedContent(1)
                        continuation.yield(
                            .init(content: try Content(raw).asPartiallyGenerated(), rawContent: raw, usage: earlier)
                        )
                        let upstream = child.streamResponse(
                            within: session,
                            to: prompt,
                            generating: type,
                            includeSchemaInPrompt: includeSchemaInPrompt,
                            options: options
                        )
                        for try await snapshot in upstream {
                            continuation.yield(snapshot)
                        }
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

/// A model whose stream yields a snapshot with the usage of an earlier request,
/// then relays the stream of a local model whose round completes
/// and whose response waits for `child.release` and then stops.
private struct LocalPrefixedCancelledModel: LanguageModel {
    typealias UnavailableReason = Never
    let earlier = LocalGenerationUsage(promptTokenCount: 30, generatedTokenCount: 5).value
    let child = LocalCancelledRoundModel(throwsCancellation: false)

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        try await streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        ).collect()
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let (earlier, child) = (earlier, child)
        return .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        let raw = GeneratedContent(kind: .array([]))
                        continuation.yield(
                            .init(content: try Content(raw).asPartiallyGenerated(), rawContent: raw, usage: earlier)
                        )
                        let upstream = child.streamResponse(
                            within: session,
                            to: prompt,
                            generating: type,
                            includeSchemaInPrompt: includeSchemaInPrompt,
                            options: options
                        )
                        for try await snapshot in upstream {
                            continuation.yield(snapshot)
                        }
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

/// A model whose own request reports usage in snapshots
/// around the snapshot of a local model's structured stream.
///
/// When `childFirst` is false, it reports `first`, waits for `relayChild`,
/// relays the structured stream, waits for `reportAgain`, and then reports `second`,
/// which includes `first`.
/// When `childFirst` is true, the structured stream completes before the model reports `first`,
/// and the model then relays the structured snapshot.
private struct LocalInterleavedModel: LanguageModel {
    typealias UnavailableReason = Never
    let childFirst: Bool
    let first = LocalGenerationUsage(promptTokenCount: 30, generatedTokenCount: 5).value
    let second = LocalGenerationUsage(promptTokenCount: 40, generatedTokenCount: 7).value
    let child = LocalSingleRoundModel()
    let relayChild = Gate()
    let reportAgain = Gate()

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        try await streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        ).collect()
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let model = self
        return .init(
            stream: AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        let raw = GeneratedContent(1)
                        let content = try Content(raw).asPartiallyGenerated()
                        func report(_ usage: LanguageModelSession.Usage) {
                            continuation.yield(.init(content: content, rawContent: raw, usage: usage))
                        }
                        let upstream = model.child.streamResponse(
                            within: session,
                            to: prompt,
                            generating: type,
                            includeSchemaInPrompt: includeSchemaInPrompt,
                            options: options
                        )
                        if model.childFirst {
                            var snapshots: [LanguageModelSession.ResponseStream<Content>.Snapshot] = []
                            for try await snapshot in upstream { snapshots.append(snapshot) }
                            report(model.first)
                            for snapshot in snapshots { continuation.yield(snapshot) }
                        } else {
                            report(model.first)
                            await model.relayChild.wait()
                            for try await snapshot in upstream { continuation.yield(snapshot) }
                            await model.reportAgain.wait()
                            report(model.second)
                        }
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
