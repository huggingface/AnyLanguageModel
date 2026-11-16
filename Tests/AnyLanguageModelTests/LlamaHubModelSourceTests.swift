import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    import HuggingFace

    /// A repository whose listing and downloads the test controls.
    final class FakeHubRepository: LlamaHubRepository, @unchecked Sendable {
        let cache: HubCache?
        let commits: [String: String]
        let listing: [String]
        /// An error that the listing throws, if any.
        let listingError: (any Error)?
        /// Whether the listing waits for cancellation and then throws `URLError(.cancelled)`,
        /// as a cancelled `URLSession` request does.
        let listingWaitsForCancellation: Bool

        private let lock = NSLock()
        private var isGateOpen: Bool
        private var gateWaiters: [CheckedContinuation<Void, Never>] = []
        private var _listingCalls = 0
        private var _downloadStarts = 0
        private var _requestedRevisions: [String] = []

        init(
            cache: HubCache? = nil,
            commits: [String: String] = ["main": String(repeating: "a", count: 40)],
            listing: [String] = ["model-Q4_K_M.gguf"],
            listingError: (any Error)? = nil,
            listingWaitsForCancellation: Bool = false,
            gateOpen: Bool = true
        ) {
            self.cache =
                cache
                ?? HubCache(
                    cacheDirectory: FileManager.default.temporaryDirectory
                        .appendingPathComponent("FakeHubRepository-\(UUID().uuidString)")
                )
            self.commits = commits
            self.listing = listing
            self.listingError = listingError
            self.listingWaitsForCancellation = listingWaitsForCancellation
            self.isGateOpen = gateOpen
        }

        var listingCalls: Int { lock.withLock { _listingCalls } }
        var downloadStarts: Int { lock.withLock { _downloadStarts } }
        var requestedRevisions: [String] { lock.withLock { _requestedRevisions } }

        /// Lets every waiting and later download finish.
        func openGate() {
            let waiters = lock.withLock {
                isGateOpen = true
                defer { gateWaiters.removeAll() }
                return gateWaiters
            }
            waiters.forEach { $0.resume() }
        }

        func commit(of repo: Repo.ID, branch: String) async throws -> String? {
            lock.withLock { _listingCalls += 1 }
            if listingWaitsForCancellation {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(5))
                }
                throw URLError(.cancelled)
            }
            if let listingError { throw listingError }
            return commits[branch]
        }

        func files(in repo: Repo.ID, revision: String) async throws -> [String] {
            lock.withLock { _requestedRevisions.append(revision) }
            return listing
        }

        func download(_ path: String, from repo: Repo.ID, revision: String) async throws -> URL {
            await withCheckedContinuation { continuation in
                let open = lock.withLock {
                    _downloadStarts += 1
                    _requestedRevisions.append(revision)
                    if !isGateOpen { gateWaiters.append(continuation) }
                    return isGateOpen
                }
                if open { continuation.resume() }
            }
            if Task.isCancelled { throw URLError(.cancelled) }
            return URL(fileURLWithPath: "/fake/\(revision)/\(path)")
        }
    }

    /// Waits until a condition holds, for at most five seconds.
    func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(5)
        while !condition() {
            guard Date() < deadline else {
                Issue.record("Timed out waiting for a condition")
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }

    @Suite("LlamaHubModelSource", .timeLimit(.minutes(1)))
    struct LlamaHubModelSourceTests {
        let commit = String(repeating: "a", count: 40)

        func makeSource(_ repository: FakeHubRepository, modelId: String = "user/repo") -> LlamaHubModelSource {
            LlamaHubModelSource(modelId: modelId, file: nil, mmproj: .automatic, hub: nil, repository: repository)
        }

        @Test func cancelledCallerDoesNotStopAnotherCallersDownload() async throws {
            let repository = FakeHubRepository(gateOpen: false)
            let source = makeSource(repository)

            let first = Task { try await source.files() }
            let second = Task { try await source.files() }
            try await waitUntil { source.waiterCount == 2 && repository.downloadStarts == 1 }

            // The cancelled caller leaves while the download is still running.
            first.cancel()
            await #expect(throws: CancellationError.self) { try await first.value }
            #expect(source.resolvedFiles == nil)

            // A caller that is already cancelled when it joins doesn't stop the download either.
            let late = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await source.files()
            }
            await #expect(throws: CancellationError.self) { try await late.value }

            repository.openGate()
            let files = try await second.value
            #expect(files == LlamaModelFiles(modelPath: "/fake/\(commit)/model-Q4_K_M.gguf", mmprojPath: nil))
            #expect(source.resolvedFiles == files)
            #expect(repository.downloadStarts == 1)
        }

        @Test func lastCancelledCallerStopsTheDownloadAndTheNextCallRetries() async throws {
            let repository = FakeHubRepository(gateOpen: false)
            let source = makeSource(repository)

            let first = Task { try await source.files() }
            try await waitUntil { repository.downloadStarts == 1 }
            first.cancel()
            await #expect(throws: CancellationError.self) { try await first.value }

            repository.openGate()
            let files = try await source.files()
            #expect(files.modelPath == "/fake/\(commit)/model-Q4_K_M.gguf")
            #expect(repository.downloadStarts == 2)
        }

        @Test func prewarmFinishesTheDownloadOfACancelledRequest() async throws {
            let repository = FakeHubRepository(gateOpen: false)
            let source = makeSource(repository)
            let model = LlamaLanguageModel(
                givenModelPath: "",
                givenMMProjPath: nil,
                hubSource: source,
                gpuLayers: 0
            )
            let session = LanguageModelSession(model: model)

            let request = Task { try await session.respond(to: "Hello") }
            model.prewarm(for: session, promptPrefix: nil)
            try await waitUntil { source.waiterCount == 2 }

            request.cancel()
            await #expect(throws: CancellationError.self) { try await request.value }

            // Prewarm still waits for the download.
            // Loading the fake file then fails, and prewarm ignores that error.
            repository.openGate()
            try await waitUntil { source.resolvedFiles != nil }
            #expect(model.localModelPath == "/fake/\(commit)/model-Q4_K_M.gguf")
            #expect(repository.downloadStarts == 1)
        }
    }
#endif  // Llama
