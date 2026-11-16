import Foundation

#if Llama
    import HuggingFace

    extension LlamaLanguageModel {
        /// How a model loaded from the Hugging Face Hub chooses its multimodal projector file.
        ///
        /// - Note: This API is exclusive to AnyLanguageModel
        ///   and using it means your code is no longer drop-in compatible
        ///   with the Foundation Models framework.
        public enum Projector: Sendable, Hashable {
            /// Selects a projector the way llama.cpp's `-hf` option does.
            ///
            /// The projector is a GGUF file whose path contains `mmproj`,
            /// in the model's directory or one of its ancestors.
            /// The deepest directory wins,
            /// then the quantization closest to the model's.
            /// A repository without a projector loads the model for text only.
            case automatic

            /// Loads the model without a projector, like llama.cpp's `--no-mmproj` option.
            case disabled

            /// Loads the projector at this path in the repository.
            case file(String)
        }

        /// An error that occurs when a model ID can't select files in a Hugging Face repository.
        ///
        /// Network and cache errors from the Hub client are thrown unchanged.
        ///
        /// - Note: This API is exclusive to AnyLanguageModel
        ///   and using it means your code is no longer drop-in compatible
        ///   with the Foundation Models framework.
        public enum HubError: Error, LocalizedError, Equatable, Sendable {
            /// The model ID isn't `<user>/<model>` with an optional `:<quantization>` tag.
            case invalidModelId(String)
            /// The repository has no GGUF model file.
            case noModelFile(repository: String)
            /// The repository has no model file with the requested quantization tag.
            case quantizationNotFound(String, repository: String, available: [String])
            /// The repository has no file at the requested path.
            case fileNotFound(String, repository: String, available: [String])
            /// The requested file is a part of a split model other than the first.
            case notFirstSplitPart(String)
            /// The repository is missing parts of a split model.
            case incompleteSplit(String, missing: [String])
            /// The repository has no projector file at the requested path.
            case projectorNotFound(String, repository: String, available: [String])
            /// The Hub client has no cache to download files into.
            case cacheDisabled
            /// The repository has no branch to load from.
            case noBranch(repository: String)

            public var errorDescription: String? {
                switch self {
                case .invalidModelId(let modelId):
                    return "Invalid model ID '\(modelId)'; expected '<user>/<model>' or '<user>/<model>:<quantization>'"
                case .noModelFile(let repository):
                    return "No GGUF model file found in repository \(repository)"
                case .quantizationNotFound(let tag, let repository, let available):
                    return "No GGUF model file with quantization '\(tag)' found in repository \(repository)"
                        + Self.listing(available)
                case .fileNotFound(let file, let repository, let available):
                    return "File '\(file)' not found in repository \(repository)" + Self.listing(available)
                case .notFirstSplitPart(let file):
                    return "File '\(file)' is not the first part of a split model"
                case .incompleteSplit(let file, let missing):
                    return "Split model '\(file)' is missing parts: \(missing.joined(separator: ", "))"
                case .projectorNotFound(let file, let repository, let available):
                    return "Projector file '\(file)' not found in repository \(repository)" + Self.listing(available)
                case .cacheDisabled:
                    return "Loading a GGUF model from the Hub requires a HubClient with a cache"
                case .noBranch(let repository):
                    return "No branch found in repository \(repository)"
                }
            }

            private static func listing(_ files: [String]) -> String {
                files.isEmpty ? "" : ". Available GGUF files: \(files.joined(separator: ", "))"
            }
        }

        /// Creates a Llama language model that loads a GGUF model from the Hugging Face Hub.
        ///
        /// The model ID uses the syntax of llama.cpp's `-hf` option,
        /// and selects the same files:
        ///
        /// ```swift
        /// // Q4_K_M, else Q8_0, else the first GGUF model file
        /// let model = LlamaLanguageModel(modelId: "ggml-org/gemma-3-4b-it-GGUF")
        ///
        /// // A quantization tag, matched without regard to case
        /// let q8 = LlamaLanguageModel(modelId: "ggml-org/gemma-3-4b-it-GGUF:Q8_0")
        /// ```
        ///
        /// The initializer doesn't access the network.
        /// The model's files download when the model is first used or prewarmed,
        /// into the Hugging Face cache that the `hf` command and llama.cpp also use.
        /// Until then, ``localModelPath`` and ``mmprojPath`` are `nil`.
        /// Later loads reuse the cached files,
        /// and use the cached snapshot when the Hub can't be reached.
        /// A split model downloads every part.
        /// Cancelling the request that is downloading cancels the download;
        /// the next load resumes it.
        ///
        /// - Parameters:
        ///   - modelId: The repository, with an optional quantization tag,
        ///     such as `"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"`.
        ///   - file: The path of a model file in the repository,
        ///     like llama.cpp's `--hf-file` option.
        ///     When set, it overrides the quantization tag.
        ///     For a split model, name the first part.
        ///   - mmproj: How to choose the multimodal projector file.
        ///     Defaults to ``Projector/automatic``.
        ///   - hub: The Hugging Face Hub client used to download files.
        ///     If not provided, the default client is used.
        ///   - gpuLayers: The number of model layers to offload to the GPU.
        ///     Defaults to ``defaultGPULayerCount``.
        public convenience init(
            modelId: String,
            file: String? = nil,
            mmproj: Projector = .automatic,
            hub: HubClient? = nil,
            gpuLayers: Int32 = LlamaLanguageModel.defaultGPULayerCount
        ) {
            self.init(
                givenModelPath: "",
                givenMMProjPath: nil,
                hubSource: LlamaHubModelSource(modelId: modelId, file: file, mmproj: mmproj, hub: hub),
                gpuLayers: gpuLayers
            )
        }

        /// The Hugging Face model ID, or `nil` for a model loaded from a file path.
        public var modelId: String? {
            hubSource?.modelId
        }

        /// The model file requested in the Hugging Face repository, if any.
        public var file: String? {
            hubSource?.file
        }

        /// How the projector file is chosen in the Hugging Face repository.
        ///
        /// For a model loaded from a file path, this is ``Projector/disabled``;
        /// use ``mmprojPath`` instead.
        public var mmproj: Projector {
            hubSource?.mmproj ?? .disabled
        }

        /// The Hugging Face Hub client used to download files,
        /// or `nil` when the default client is used or the model loads from a file path.
        public var hub: HubClient? {
            hubSource?.hub
        }

        /// Resolves, downloads, and loads the model.
        ///
        /// A later request uses the loaded model.
        /// Errors are ignored here; the next request throws them.
        public func prewarm(
            for session: LanguageModelSession,
            promptPrefix: Prompt?
        ) {
            Task {
                try? await ensureModelLoaded()
            }
        }
    }

    /// The local paths of a model's files.
    struct LlamaModelFiles: Equatable, Sendable {
        var modelPath: String
        var mmprojPath: String?
    }

    /// The Hub operations that loading a model needs.
    ///
    /// ``LlamaHubModelSource`` uses a `HubClient` through this protocol,
    /// and tests use it to control listings and downloads without the network.
    protocol LlamaHubRepository: Sendable {
        /// The cache that downloads go into, or `nil` if caching is disabled.
        var cache: HubCache? { get }

        /// Returns the names of the repository's branches, in the Hub's order.
        func branches(of repo: Repo.ID) async throws -> [String]

        /// Returns the commit that a branch points to, if the Hub reports one.
        func commit(of repo: Repo.ID, branch: String) async throws -> String?

        /// Returns the paths of every file in the repository at a revision, in the Hub's order.
        func files(in repo: Repo.ID, revision: String) async throws -> [String]

        /// Downloads a file into the cache and returns its path in the snapshot directory.
        func download(_ path: String, from repo: Repo.ID, revision: String) async throws -> URL
    }

    /// The Hub operations, performed by a `HubClient`.
    struct HubClientRepository: LlamaHubRepository {
        let hub: HubClient

        var cache: HubCache? { hub.cache }

        func branches(of repo: Repo.ID) async throws -> [String] {
            try await hub.modelRefs(repo).branches.map(\.name)
        }

        // `Git.Ref.targetOid` is always nil in swift-huggingface 1.0.0:
        // it decodes `targetOid`, but the Hub's refs response has `targetCommit`.
        // The model info at the branch has the commit instead.
        func commit(of repo: Repo.ID, branch: String) async throws -> String? {
            try await hub.getModel(repo, revision: branch).sha
        }

        func files(in repo: Repo.ID, revision: String) async throws -> [String] {
            var paths: [String] = []
            for try await page in try await hub.listAllTree(in: repo, revision: revision, recursive: true) {
                paths += page.items.filter { $0.type == .file }.map(\.path)
            }
            return paths
        }

        func download(_ path: String, from repo: Repo.ID, revision: String) async throws -> URL {
            try await hub.downloadFile(at: path, from: repo, revision: revision)
        }
    }

    /// Resolves and downloads the files for a Hugging Face model ID.
    ///
    /// Concurrent callers share one download.
    /// A cancelled caller stops waiting at once and throws `CancellationError`;
    /// the download continues while another caller waits for it,
    /// and stops when no caller is left.
    /// A failed or abandoned download is discarded so that the next call retries it.
    final class LlamaHubModelSource: @unchecked Sendable {
        let modelId: String
        let file: String?
        let mmproj: LlamaLanguageModel.Projector
        let hub: HubClient?

        /// The repository to use instead of `hub`, for tests.
        private let repository: (any LlamaHubRepository)?

        private let lock = NSLock()
        /// The running download, protected by `lock`.
        private var task: Task<Void, Never>?
        /// Identifies the running download, so that an abandoned download's result is ignored.
        private var generation = 0
        /// The callers waiting for the running download, protected by `lock`.
        private var waiters: [UUID: CheckedContinuation<LlamaModelFiles, any Error>] = [:]
        private var resolved: LlamaModelFiles?

        init(
            modelId: String,
            file: String?,
            mmproj: LlamaLanguageModel.Projector,
            hub: HubClient?,
            repository: (any LlamaHubRepository)? = nil
        ) {
            self.modelId = modelId
            self.file = file
            self.mmproj = mmproj
            self.hub = hub
            self.repository = repository
        }

        /// The local paths, once a resolution has succeeded.
        var resolvedFiles: LlamaModelFiles? {
            lock.withLock { resolved }
        }

        /// The number of callers waiting for the running download.
        var waiterCount: Int {
            lock.withLock { waiters.count }
        }

        /// Returns the local paths, downloading files that aren't cached yet.
        ///
        /// A caller that is cancelled, or is already cancelled when it calls,
        /// throws `CancellationError` without waiting.
        /// It cancels the shared download only if no other caller is waiting for it.
        func files() async throws -> LlamaModelFiles {
            let id = UUID()
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    lock.withLock {
                        if let resolved {
                            continuation.resume(returning: resolved)
                            return
                        }
                        if Task.isCancelled {
                            continuation.resume(throwing: CancellationError())
                            return
                        }
                        waiters[id] = continuation
                        if task == nil {
                            generation += 1
                            let generation = generation
                            task = Task { await self.run(generation: generation) }
                        }
                    }
                }
            } onCancel: {
                let (continuation, abandoned) = lock.withLock {
                    () -> (CheckedContinuation<LlamaModelFiles, any Error>?, Task<Void, Never>?) in
                    guard let continuation = waiters.removeValue(forKey: id) else {
                        return (nil, nil)
                    }
                    guard waiters.isEmpty, let task else {
                        return (continuation, nil)
                    }
                    // No caller is left, so the next caller starts a new download.
                    self.task = nil
                    generation += 1
                    return (continuation, task)
                }
                continuation?.resume(throwing: CancellationError())
                abandoned?.cancel()
            }
        }

        /// Runs a download and passes its result to every waiting caller.
        private func run(generation: Int) async {
            let result: Result<LlamaModelFiles, any Error>
            do {
                result = .success(try await resolveAndDownload())
            } catch {
                result = .failure(error)
            }

            let continuations = lock.withLock { () -> [CheckedContinuation<LlamaModelFiles, any Error>] in
                guard self.generation == generation else { return [] }
                if case .success(let files) = result {
                    resolved = files
                }
                task = nil
                defer { waiters.removeAll() }
                return Array(waiters.values)
            }
            for continuation in continuations {
                continuation.resume(with: result)
            }
        }

        /// Resolves the model ID and downloads the selected files.
        func resolveAndDownload() async throws -> LlamaModelFiles {
            let (repository, tag) = try LlamaHubFileResolver.parseModelId(modelId)
            let hub = self.repository ?? HubClientRepository(hub: self.hub ?? HubClient.default)
            guard let cache = hub.cache else {
                throw LlamaLanguageModel.HubError.cacheDisabled
            }
            guard let repo = Repo.ID(rawValue: repository) else {
                throw LlamaLanguageModel.HubError.invalidModelId(modelId)
            }

            // Like llama.cpp, use `main` or else the first branch,
            // list the files at the commit that the branch points to,
            // and download them at that commit,
            // so the parts of a split model come from one revision.
            let branch: String
            let revision: String
            let listing: [String]
            do {
                guard let selected = LlamaHubFileResolver.selectBranch(try await hub.branches(of: repo)) else {
                    throw LlamaLanguageModel.HubError.noBranch(repository: repository)
                }
                branch = selected
                revision = try await hub.commit(of: repo, branch: branch) ?? branch
                listing = try await hub.files(in: repo, revision: revision)
            } catch {
                // A cancelled URLSession request throws `URLError(.cancelled)`,
                // not `CancellationError`, so check the task too.
                // A cancelled load must not fall back to the cache.
                if error is CancellationError || Task.isCancelled {
                    throw CancellationError()
                }
                if let cached = try? cachedFiles(repo: repo, repository: repository, tag: tag, cache: cache) {
                    return cached
                }
                throw error
            }

            let selection = try LlamaHubFileResolver.select(
                from: listing,
                repository: repository,
                tag: tag,
                file: file,
                mmproj: mmproj
            )

            var modelURLs: [URL] = []
            for path in selection.modelFiles {
                modelURLs.append(try await hub.download(path, from: repo, revision: revision))
            }
            let projectorURL: URL? =
                if let projector = selection.projectorFile {
                    try await hub.download(projector, from: repo, revision: revision)
                } else {
                    nil
                }

            // A download at a commit doesn't record which branch it came from.
            // Point the branch's ref at it, as llama.cpp and huggingface_hub do,
            // so that an offline load finds this snapshot.
            if revision != branch {
                try? cache.updateRef(repo: repo, kind: .model, ref: branch, commit: revision)
            }

            return LlamaModelFiles(modelPath: modelURLs[0].path, mmprojPath: projectorURL?.path)
        }

        /// Selects files from the cached snapshot that `refs/main`,
        /// or else another branch's ref, points to.
        ///
        /// Files are listed in path order, because a directory listing has no defined order.
        private func cachedFiles(
            repo: Repo.ID,
            repository: String,
            tag: String?,
            cache: HubCache
        ) throws -> LlamaModelFiles? {
            let refs = cache.refsDirectory(repo: repo, kind: .model)
            guard let commit = LlamaHubFileResolver.cachedCommit(inRefsDirectory: refs) else {
                return nil
            }
            let snapshot = try cache.snapshotPath(repo: repo, kind: .model, commitHash: commit)
            guard let enumerator = FileManager.default.enumerator(atPath: snapshot.path) else {
                return nil
            }

            // Snapshot entries are symbolic links to blobs; `fileExists` follows them.
            var listing: [String] = []
            for case let path as String in enumerator {
                var isDirectory: ObjCBool = false
                let exists = FileManager.default.fileExists(
                    atPath: snapshot.appendingPathComponent(path).path,
                    isDirectory: &isDirectory
                )
                if exists && !isDirectory.boolValue {
                    listing.append(path)
                }
            }
            listing.sort()

            let selection = try LlamaHubFileResolver.select(
                from: listing,
                repository: repository,
                tag: tag,
                file: file,
                mmproj: mmproj
            )
            return LlamaModelFiles(
                modelPath: snapshot.appendingPathComponent(selection.modelFiles[0]).path,
                mmprojPath: selection.projectorFile.map { snapshot.appendingPathComponent($0).path }
            )
        }
    }
#endif  // Llama
