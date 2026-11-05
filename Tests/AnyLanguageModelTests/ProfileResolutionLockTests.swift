import Dispatch
import Testing

@testable import AnyLanguageModel

@Suite("Profile resolution lock")
struct ProfileResolutionLockTests {
    // Both resolutions block in a history transform,
    // so they run outside the cooperative pool.
    @available(macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test func historyTransformsOfOverlappingResolutionsDontRunAtTheSameTime() async {
        let probe = TransformOverlapProbe()
        let session = LanguageModelSession(profile: TransformOverlapProfile(probe: probe))

        let first = Task.detached(executorPreference: BlockingTaskExecutor()) {
            _ = session.resolvedRequestContext()
        }
        await probe.waitUntilTheFirstTransformStarts()
        let second = Task.detached(executorPreference: BlockingTaskExecutor()) {
            _ = session.resolvedRequestContext()
        }
        await first.value
        await second.value

        #expect(probe.counts == .init(calls: 2, maximumAtOnce: 1))
    }
}

/// Records how many history transforms run at the same time.
///
/// The first transform waits for a short time for a second transform to start,
/// which can happen only if transforms run outside the profile lock.
private final class TransformOverlapProbe: Sendable {
    struct Counts: Equatable {
        var calls = 0
        var maximumAtOnce = 0
    }

    private let state = Locked((counts: Counts(), running: 0))
    private let started: AsyncStream<Void>
    private let startedContinuation: AsyncStream<Void>.Continuation
    private let secondStarted = DispatchSemaphore(value: 0)

    init() {
        (started, startedContinuation) = AsyncStream<Void>.makeStream()
    }

    var counts: Counts {
        state.withLock { $0.counts }
    }

    func transform(_ entries: [Transcript.Entry]) -> [Transcript.Entry] {
        let call = state.withLock { state in
            state.running += 1
            state.counts.calls += 1
            state.counts.maximumAtOnce = max(state.counts.maximumAtOnce, state.running)
            return state.counts.calls
        }
        if call == 1 {
            startedContinuation.yield()
            _ = secondStarted.wait(timeout: .now() + .milliseconds(500))
        } else {
            secondStarted.signal()
        }
        state.withLock { $0.running -= 1 }
        return entries
    }

    func waitUntilTheFirstTransformStarts() async {
        for await _ in started.prefix(1) {}
    }
}

private struct TransformOverlapProfile: LanguageModelSession.DynamicProfile {
    let probe: TransformOverlapProbe

    var body: some LanguageModelSession.DynamicProfile {
        LanguageModelSession.Profile { Instructions("Transforms") }
            .historyTransform { probe.transform($0) }
    }
}
