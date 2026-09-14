import Foundation
import Observation
import Testing

@testable import AnyLanguageModel

@Suite("Response token usage")
struct UsageTests {
    typealias Usage = LanguageModelSession.Usage

    private let usage = Usage(
        input: .init(totalTokenCount: 100, cachedTokenCount: 25),
        output: .init(totalTokenCount: 20, reasoningTokenCount: 5)
    )

    @Test func emptyAndZeroCounts() {
        #expect(ReportedUsage().isEmpty)
        #expect(ReportedUsage().normalized == nil)
        let zero = ReportedUsage(input: .init(totalTokenCount: 0))
        #expect(!zero.isEmpty)
        #expect(zero.normalized?.input.totalTokenCount == 0)
    }

    @Test func addsSeparateRoundsAndMergesCumulativeUpdates() {
        var total = ReportedUsage()
        total.add(nil)
        #expect(total.isEmpty)
        total.add(
            .init(
                input: .init(totalTokenCount: 100, cachedTokenCount: 25),
                output: .init(totalTokenCount: 20, reasoningTokenCount: 5)
            )
        )
        total.add(.init(input: .init(totalTokenCount: 10), output: .init(totalTokenCount: 3)))
        #expect(total.input.totalTokenCount == 110)
        #expect(total.input.cachedTokenCount == 25)
        #expect(total.output.totalTokenCount == 23)
        #expect(total.output.reasoningTokenCount == 5)

        total.merge(.init(input: .init(cachedTokenCount: 0), output: .init(totalTokenCount: 7)))
        total.merge(nil)
        #expect(total.input.totalTokenCount == 110)
        #expect(total.input.cachedTokenCount == 0)
        #expect(total.output.totalTokenCount == 7)
        #expect(total.output.reasoningTokenCount == 5)

        var partial = ReportedUsage()
        partial.add(.init(output: .init(totalTokenCount: 2)))
        partial.add(.init(output: .init(totalTokenCount: 3)))
        #expect(partial.input.totalTokenCount == nil)
        #expect(partial.output.reasoningTokenCount == nil)
        #expect(partial.output.totalTokenCount == 5)
    }

    @Test func reportedMetadataSurvivesNormalizationAndUpdates() {
        var reported = ReportedUsage(metadata: ["cache_creation_input_tokens": GeneratedContent(10)])
        #expect(!reported.isEmpty)
        #expect(reported.normalized?.value.metadata["cache_creation_input_tokens"] == GeneratedContent(10))
        #expect(reported.value.totalTokenCount == 0)

        reported.merge(.init(output: .init(totalTokenCount: 7)))
        #expect(reported.value.metadata["cache_creation_input_tokens"] == GeneratedContent(10))
        reported.merge(.init(metadata: ["cache_creation_input_tokens": GeneratedContent(0)]))
        #expect(reported.value.metadata["cache_creation_input_tokens"] == GeneratedContent(0))

        reported.add(.init(metadata: ["cache_creation_input_tokens": GeneratedContent(5)]))
        reported.add(.init(output: .init(totalTokenCount: 3)))
        #expect(reported.value.metadata["cache_creation_input_tokens"] == GeneratedContent(5))
        #expect(reported.value.output.totalTokenCount == 10)
    }

    @Test func streamingIncrementsOnlyIncludeChangedMetadata() {
        var previous = Usage.zero
        var snapshot = usage
        snapshot.metadata = [
            "cache_creation_input_tokens": GeneratedContent(10),
            "service_tier": GeneratedContent("standard"),
        ]
        #expect(snapshot.increment(since: &previous) == snapshot)
        #expect(previous == snapshot)
        #expect(snapshot.increment(since: &previous) == .zero)

        snapshot.metadata["service_tier"] = GeneratedContent("priority")
        let changed = snapshot.increment(since: &previous)
        #expect(changed.totalTokenCount == 0)
        #expect(changed.metadata == ["service_tier": GeneratedContent("priority")])
        #expect(previous == snapshot)
        #expect(snapshot.increment(since: &previous) == .zero)

        let retainedMetadata = snapshot.metadata
        snapshot.metadata = [:]
        #expect(snapshot.increment(since: &previous) == .zero)
        #expect(previous.metadata == retainedMetadata)
        snapshot.metadata = retainedMetadata
        #expect(snapshot.increment(since: &previous) == .zero)

        snapshot.metadata["cache_creation_input_tokens"] = GeneratedContent(0)
        snapshot.output.totalTokenCount += 3
        let increased = snapshot.increment(since: &previous)
        #expect(increased.output.totalTokenCount == 3)
        #expect(increased.metadata == ["cache_creation_input_tokens": GeneratedContent(0)])
        #expect(snapshot.increment(since: &previous) == .zero)
    }

    @Test func codableRoundTrip() throws {
        var withMetadata = usage
        withMetadata.metadata = ["service_tier": GeneratedContent("standard")]
        for value in [usage, .zero, withMetadata] {
            let data = try JSONEncoder().encode(value)
            #expect(try JSONDecoder().decode(Usage.self, from: data) == value)
        }
    }

    @Test func responseAndStreamPreserveUsage() async throws {
        var model = MockLanguageModel.fixed("Hello")
        model.usage = usage
        let session = LanguageModelSession(model: model)
        #expect(session.usage == .zero)
        #expect(try await session.respond(to: "Hi").usage == usage)
        #expect(session.usage == usage)
        for try await snapshot in session.streamResponse(to: "Hi") {
            #expect(snapshot.usage == usage)
        }
        let collected = try await session.streamResponse(to: "Hi").collect()
        #expect(collected.content == "Hello")
        #expect(collected.usage == usage)
        #expect(session.usage.input.totalTokenCount == 300)
        #expect(session.usage.input.cachedTokenCount == 75)
        #expect(session.usage.output.totalTokenCount == 60)
        #expect(session.usage.output.reasoningTokenCount == 15)
        #expect(session.usage.totalTokenCount == 360)
    }

    @Test func fallbackAndExistingInitializers() async throws {
        let stream = LanguageModelSession.ResponseStream(
            content: "Hello",
            rawContent: GeneratedContent("Hello"),
            usage: usage
        )
        #expect(try await stream.collect().usage == usage)
        for try await snapshot in stream {
            #expect(snapshot.usage == usage)
        }
        let response = LanguageModelSession.Response(
            content: "Hello",
            rawContent: GeneratedContent("Hello"),
            transcriptEntries: []
        )
        let snapshot = LanguageModelSession.ResponseStream<String>.Snapshot(
            content: "Hello",
            rawContent: GeneratedContent("Hello")
        )
        let fallback = LanguageModelSession.ResponseStream(content: "Hello", rawContent: GeneratedContent("Hello"))
        #expect(response.usage == .zero)
        #expect(snapshot.usage == .zero)
        #expect(try await fallback.collect().usage == .zero)
    }
    @Test func metadataAndCombinedTotal() throws {
        var value = Usage(
            input: .init(totalTokenCount: 100, cachedTokenCount: 25),
            output: .init(totalTokenCount: 20, reasoningTokenCount: 5),
            metadata: ["cache_creation_input_tokens": 10, "service_tier": "standard"]
        )
        #expect(value.totalTokenCount == 120)
        #expect(value.metadata["cache_creation_input_tokens"] == GeneratedContent(10))
        #expect(value.metadata["service_tier"] == GeneratedContent("standard"))
        value.input.totalTokenCount = 150
        value.output.totalTokenCount = 30
        #expect(value.totalTokenCount == 180)
        value.metadata["service_tier"] = GeneratedContent("priority")
        #expect(try JSONDecoder().decode(Usage.self, from: JSONEncoder().encode(value)) == value)
    }

    @Test func imageResponsesAccumulateUsage() async throws {
        var model = MockLanguageModel.fixed("Hello")
        model.usage = usage
        let session = LanguageModelSession(model: model)
        let image = Transcript.ImageSegment(url: URL(string: "https://example.com/image.png")!)
        let response = try await session.respond(to: "Hi", image: image, generating: String.self)
        #expect(response.usage == usage)
        #expect(session.usage == usage)
        let streamed = try await session.streamResponse(to: "Hi", image: image, generating: String.self).collect()
        #expect(streamed.usage == usage)
        #expect(session.usage.totalTokenCount == 240)
    }

    @Test func restoredTranscriptStartsWithZeroUsage() async throws {
        var model = MockLanguageModel.fixed("Hello")
        model.usage = usage
        let original = LanguageModelSession(model: model)
        try await original.respond(to: "Hi")
        let restored = LanguageModelSession(model: model, transcript: original.transcript)
        #expect(restored.usage == .zero)
        try await restored.respond(to: "Again")
        #expect(restored.usage == usage)
    }

    @Test func sessionUsageIsObservable() async throws {
        var model = MockLanguageModel.fixed("Hello")
        model.usage = usage
        let session = LanguageModelSession(model: model)
        let changed = Locked(false)
        withObservationTracking {
            _ = session.usage
        } onChange: {
            changed.withLock { $0 = true }
        }
        try await session.respond(to: "Hi")
        #expect(changed.withLock { $0 })
        #expect(session.usage == usage)
    }

    @Test func streamingCountsAreMonotonicAndSurviveFailures() async throws {
        let session = LanguageModelSession(model: UsageStreamModel(fail: false))
        for try await snapshot in session.streamResponse(to: "Hi") {
            #expect(session.usage.totalTokenCount >= snapshot.usage.totalTokenCount)
        }
        #expect(session.usage.input.totalTokenCount == 100)
        #expect(session.usage.output.totalTokenCount == 21)
        #expect(session.usage.metadata["service_tier"] == GeneratedContent("priority"))
        let response = try await session.streamResponse(to: "Again").collect()
        #expect(response.usage.totalTokenCount == 121)
        #expect(session.usage.totalTokenCount == 242)

        let failing = LanguageModelSession(model: UsageStreamModel(fail: true))
        await #expect(throws: UsageStreamModel.StreamFailure.self) {
            try await failing.streamResponse(to: "Hi").collect()
        }
        #expect(failing.usage.totalTokenCount == 121)
    }

}

private struct UsageStreamModel: LanguageModel {
    typealias UnavailableReason = Never
    var availability: Availability<Never> { .available }
    let fail: Bool

    struct StreamFailure: Error {}

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
        .init(
            stream: AsyncThrowingStream { continuation in
                do {
                    let raw = GeneratedContent("Hello")
                    let content = try Content(raw)
                    for count in [2, 20, 20, 15, 21] {
                        continuation.yield(
                            .init(
                                content: content.asPartiallyGenerated(),
                                rawContent: raw,
                                usage: .init(
                                    input: .init(totalTokenCount: 100, cachedTokenCount: 25),
                                    output: .init(totalTokenCount: count, reasoningTokenCount: 0),
                                    metadata: ["service_tier": "priority"]
                                )
                            )
                        )
                    }
                    if fail { continuation.finish(throwing: StreamFailure()) } else { continuation.finish() }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        )
    }
}
