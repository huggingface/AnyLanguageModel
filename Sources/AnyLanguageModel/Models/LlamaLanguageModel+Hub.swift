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
                localModelPath: "",
                localMMProjPath: nil,
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

    /// Resolves and downloads the files for a Hugging Face model ID.
    ///
    /// Concurrent callers share one resolution.
    /// A failed resolution is discarded so that the next call retries it.
    final class LlamaHubModelSource: @unchecked Sendable {
        let modelId: String
        let file: String?
        let mmproj: LlamaLanguageModel.Projector
        let hub: HubClient?

        private let lock = NSLock()
        private var task: Task<LlamaModelFiles, any Error>?
        private var resolved: LlamaModelFiles?

        init(modelId: String, file: String?, mmproj: LlamaLanguageModel.Projector, hub: HubClient?) {
            self.modelId = modelId
            self.file = file
            self.mmproj = mmproj
            self.hub = hub
        }

        /// The local paths, once a resolution has succeeded.
        var resolvedFiles: LlamaModelFiles? {
            lock.withLock { resolved }
        }

        /// Returns the local paths, downloading files that aren't cached yet.
        ///
        /// Cancelling the caller cancels the shared download.
        func files() async throws -> LlamaModelFiles {
            let task = lock.withLock {
                if let task = self.task {
                    return task
                }
                let task = Task { try await self.download() }
                self.task = task
                return task
            }

            do {
                let files = try await withTaskCancellationHandler {
                    try await task.value
                } onCancel: {
                    task.cancel()
                }
                lock.withLock { resolved = files }
                return files
            } catch {
                lock.withLock {
                    if self.task == task {
                        self.task = nil
                    }
                }
                throw error
            }
        }

        private func download() async throws -> LlamaModelFiles {
            let (repository, tag) = try LlamaHubFileResolver.parseModelId(modelId)
            let hub = self.hub ?? HubClient.default
            guard let cache = hub.cache else {
                throw LlamaLanguageModel.HubError.cacheDisabled
            }
            guard let repo = Repo.ID(rawValue: repository) else {
                throw LlamaLanguageModel.HubError.invalidModelId(modelId)
            }

            // Like llama.cpp, list the files at the commit that `main` points to,
            // and download them at that commit,
            // so the parts of a split model come from one revision.
            let revision: String
            let listing: [String]
            do {
                revision = try await hub.getModel(repo, revision: "main").sha ?? "main"
                var paths: [String] = []
                for try await page in try await hub.listAllTree(in: repo, revision: revision, recursive: true) {
                    paths += page.items.filter { $0.type == .file }.map(\.path)
                }
                listing = paths
            } catch {
                if error is CancellationError { throw error }
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
                modelURLs.append(try await hub.downloadFile(at: path, from: repo, revision: revision))
            }
            let projectorURL: URL? =
                if let projector = selection.projectorFile {
                    try await hub.downloadFile(at: projector, from: repo, revision: revision)
                } else {
                    nil
                }

            // A download at a commit doesn't record which branch it came from.
            // Point `refs/main` at it, as llama.cpp and huggingface_hub do,
            // so that an offline load finds this snapshot.
            if revision != "main" {
                try? cache.updateRef(repo: repo, kind: .model, ref: "main", commit: revision)
            }

            return LlamaModelFiles(modelPath: modelURLs[0].path, mmprojPath: projectorURL?.path)
        }

        /// Selects files from the cached snapshot that `refs/main` points to.
        ///
        /// Files are listed in path order, because a directory listing has no defined order.
        private func cachedFiles(
            repo: Repo.ID,
            repository: String,
            tag: String?,
            cache: HubCache
        ) throws -> LlamaModelFiles? {
            guard let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main") else {
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
