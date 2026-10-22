import Foundation
import Testing

@testable import AnyLanguageModel

/// Collecting a streamed response that's cancelled throws `CancellationError`, as `respond` does:
/// not `noSnapshots`, and not the snapshots so far as if they were the whole response.
@Suite("Collecting a cancelled response")
struct CollectCancellationTests {
    @Test func cancelledBeforeTheFirstSnapshot() async throws {
        let (stream, continuation) = AsyncThrowingStream<
            LanguageModelSession.ResponseStream<String>.Snapshot, any Error
        >.makeStream()
        let collecting = Task { try await LanguageModelSession.ResponseStream<String>(stream: stream).collect() }
        collecting.cancel()
        await #expect(throws: CancellationError.self) { try await collecting.value }
        continuation.finish()
    }

    @Test func cancelledAfterTheFirstSnapshot() async throws {
        // Nothing is buffered: a yield that isn't dropped reached `collect()`.
        let (stream, continuation) = AsyncThrowingStream<
            LanguageModelSession.ResponseStream<String>.Snapshot, any Error
        >
        .makeStream(bufferingPolicy: .bufferingNewest(0))
        let collecting = Task { try await LanguageModelSession.ResponseStream<String>(stream: stream).collect() }
        let raw = GeneratedContent("Partial")
        let snapshot = LanguageModelSession.ResponseStream<String>.Snapshot(
            content: try String(raw).asPartiallyGenerated(),
            rawContent: raw
        )
        while case .dropped = continuation.yield(snapshot) { await Task.yield() }
        collecting.cancel()
        await #expect(throws: CancellationError.self) { try await collecting.value }
        continuation.finish()
    }
}
