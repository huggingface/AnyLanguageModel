import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    import HuggingFace

    @Suite("LlamaHubFileResolver")
    struct LlamaHubFileResolverTests {
        typealias HubError = LlamaLanguageModel.HubError

        // The file list of ggml-org/gemma-3-4b-it-GGUF.
        let gemma = [
            ".gitattributes",
            "README.md",
            "gemma-3-4b-it-Q4_K_M.gguf",
            "gemma-3-4b-it-Q8_0.gguf",
            "gemma-3-4b-it-f16.gguf",
            "mmproj-model-f16.gguf",
        ]

        func select(
            _ files: [String],
            tag: String? = nil,
            file: String? = nil,
            mmproj: LlamaLanguageModel.Projector = .automatic
        ) throws -> LlamaHubFileSelection {
            try LlamaHubFileResolver.select(
                from: files,
                repository: "user/repo",
                tag: tag,
                file: file,
                mmproj: mmproj
            )
        }

        @Test func parsesModelIds() throws {
            #expect(try LlamaHubFileResolver.parseModelId("user/repo") == ("user/repo", nil))
            #expect(try LlamaHubFileResolver.parseModelId("user/repo:Q8_0") == ("user/repo", "Q8_0"))
            #expect(try LlamaHubFileResolver.parseModelId("user/repo:") == ("user/repo", nil))

            for invalid in ["repo", "user/repo/extra", "/repo", "user/", "user/repo:Q4:Q8"] {
                #expect(throws: HubError.invalidModelId(invalid)) {
                    try LlamaHubFileResolver.parseModelId(invalid)
                }
            }
        }

        @Test func defaultsToQ4KM() throws {
            let selection = try select(gemma)
            #expect(selection.modelFiles == ["gemma-3-4b-it-Q4_K_M.gguf"])
        }

        @Test func defaultFallsBackToQ8ThenFirstModel() throws {
            let q8 = try select(["model-F16.gguf", "model-Q8_0.gguf"])
            #expect(q8.modelFiles == ["model-Q8_0.gguf"])

            let first = try select(["mmproj-F16.gguf", "model-imatrix.gguf", "model-F16.gguf", "model-Q6_K.gguf"])
            #expect(first.modelFiles == ["model-F16.gguf"])
        }

        @Test func selectsExplicitTag() throws {
            let selection = try select(gemma, tag: "Q8_0")
            #expect(selection.modelFiles == ["gemma-3-4b-it-Q8_0.gguf"])
        }

        @Test func matchesTagsWithoutRegardToCase() throws {
            #expect(try select(gemma, tag: "q8_0").modelFiles == ["gemma-3-4b-it-Q8_0.gguf"])
            #expect(try select(gemma, tag: "F16").modelFiles == ["gemma-3-4b-it-f16.gguf"])
        }

        @Test func matchesTagsAsWholeTokens() throws {
            let files = ["model-BF16.gguf", "model-F16.gguf", "model-Q4_K_M.gguf", "model-Q4_K.gguf"]
            #expect(try select(files, tag: "F16").modelFiles == ["model-F16.gguf"])
            #expect(try select(files, tag: "Q4_K").modelFiles == ["model-Q4_K.gguf"])
            // The tag is a literal, so `.` doesn't match any character.
            #expect(throws: HubError.self) { try select(["model-Q4_0.gguf"], tag: "Q4.0") }
        }

        @Test func missingTagThrowsWithoutFallback() throws {
            #expect(
                throws: HubError.quantizationNotFound(
                    "Q2_K",
                    repository: "user/repo",
                    available: [
                        "gemma-3-4b-it-Q4_K_M.gguf", "gemma-3-4b-it-Q8_0.gguf", "gemma-3-4b-it-f16.gguf",
                        "mmproj-model-f16.gguf",
                    ]
                )
            ) {
                try select(gemma, tag: "Q2_K")
            }
        }

        @Test func repositoryWithoutGGUFThrows() throws {
            #expect(throws: HubError.noModelFile(repository: "user/repo")) {
                try select(["README.md", "config.json", "model.safetensors"])
            }
            // A projector alone isn't a model.
            #expect(throws: HubError.noModelFile(repository: "user/repo")) {
                try select(["mmproj-model-f16.gguf"])
            }
        }

        @Test func skipsPathsOutsideTheSnapshot() throws {
            let files = [
                "../escape-Q4_K_M.gguf",
                "/absolute-Q4_K_M.gguf",
                "dir//empty-Q4_K_M.gguf",
                "model-Q4_K_M.gguf",
                "../mmproj-F16.gguf",
            ]
            let selection = try select(files)
            #expect(selection.modelFiles == ["model-Q4_K_M.gguf"])
            #expect(selection.projectorFile == nil)

            #expect(throws: HubError.self) {
                try select(files, file: "../escape-Q4_K_M.gguf")
            }
            #expect(LlamaHubFileResolver.isValidRelativePath("dir/model.gguf"))
            #expect(!LlamaHubFileResolver.isValidRelativePath("dir/../model.gguf"))
            #expect(!LlamaHubFileResolver.isValidRelativePath("dir\\model.gguf"))
        }

        @Test func explicitFileOverridesTag() throws {
            let selection = try select(gemma, tag: "Q4_K_M", file: "gemma-3-4b-it-f16.gguf")
            #expect(selection.modelFiles == ["gemma-3-4b-it-f16.gguf"])

            #expect(throws: HubError.self) {
                try select(gemma, file: "gemma-3-4b-it-Q2_K.gguf")
            }
        }

        // The split layout of unsloth/Qwen3-30B-A3B-GGUF.
        let split = [
            "BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf",
            "BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf",
            "Qwen3-30B-A3B-Q4_K_M.gguf",
            "Q8_0/Qwen3-30B-A3B-Q8_0-00002-of-00003.gguf",
            "Q8_0/Qwen3-30B-A3B-Q8_0-00001-of-00003.gguf",
            "Q8_0/Qwen3-30B-A3B-Q8_0-00003-of-00003.gguf",
        ]

        @Test func selectsEveryPartOfSplitModelInOrder() throws {
            #expect(
                try select(split, tag: "bf16").modelFiles == [
                    "BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf",
                    "BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf",
                ]
            )
            #expect(
                try select(split, tag: "Q8_0").modelFiles == [
                    "Q8_0/Qwen3-30B-A3B-Q8_0-00001-of-00003.gguf",
                    "Q8_0/Qwen3-30B-A3B-Q8_0-00002-of-00003.gguf",
                    "Q8_0/Qwen3-30B-A3B-Q8_0-00003-of-00003.gguf",
                ]
            )
            #expect(
                try select(split, file: "BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf").modelFiles.count == 2
            )
        }

        @Test func rejectsLaterSplitPartsAndMissingParts() throws {
            #expect(throws: HubError.notFirstSplitPart("BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf")) {
                try select(split, file: "BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf")
            }

            let incomplete = split.filter { !$0.contains("Q8_0-00002") }
            #expect(
                throws: HubError.incompleteSplit(
                    "Q8_0/Qwen3-30B-A3B-Q8_0-00001-of-00003.gguf",
                    missing: ["Q8_0/Qwen3-30B-A3B-Q8_0-00002-of-00003.gguf"]
                )
            ) {
                try select(incomplete, tag: "Q8_0")
            }
        }

        @Test func parsesSplitInfo() {
            #expect(
                LlamaHubFileResolver.splitInfo("dir/Model-Q4_K_M-00002-of-00003.gguf")
                    == .init(prefix: "dir/Model-Q4_K_M", tag: "Q4_K_M", index: 2, count: 3)
            )
            #expect(
                LlamaHubFileResolver.splitInfo("model.f16.gguf")
                    == .init(prefix: "model.f16", tag: "F16", index: 1, count: 1)
            )
            #expect(LlamaHubFileResolver.splitInfo("README.md") == nil)
            #expect(LlamaHubFileResolver.quantizationBits("model-Q4_K_M.gguf") == 4)
            #expect(LlamaHubFileResolver.quantizationBits("model-BF16.gguf") == 16)
            #expect(LlamaHubFileResolver.quantizationBits("model-IQ2_XXS.gguf") == 2)
        }

        @Test func selectsProjectorWithClosestQuantization() throws {
            let files = [
                "Model-Q8_0.gguf",
                "Model-F16.gguf",
                "mmproj-Model-F16.gguf",
                "mmproj-Model-Q8_0.gguf",
            ]
            #expect(try select(files).projectorFile == "mmproj-Model-Q8_0.gguf")
            #expect(try select(files, tag: "F16").projectorFile == "mmproj-Model-F16.gguf")
            // A tie keeps the first projector in listing order.
            #expect(
                try select(["Model-Q4_K_M.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"]).projectorFile
                    == "mmproj-F16.gguf"
            )
        }

        @Test func prefersProjectorInDeepestSharedDirectory() throws {
            let files = [
                "mmproj-F16.gguf",
                "BF16/Model-BF16-00001-of-00002.gguf",
                "BF16/Model-BF16-00002-of-00002.gguf",
                "BF16/mmproj-Q8_0.gguf",
                "Q8_0/mmproj-Q8_0.gguf",
                "Q8_0/Model-Q8_0.gguf",
            ]
            // The BF16 directory's projector wins despite its quantization.
            #expect(try select(files, tag: "BF16").projectorFile == "BF16/mmproj-Q8_0.gguf")
            // A projector in another directory is never chosen.
            #expect(try select(["Q8_0/Model-Q8_0.gguf", "F16/mmproj-F16.gguf"]).projectorFile == nil)
            // A model in a subdirectory can use a projector at the root.
            #expect(try select(["Q8_0/Model-Q8_0.gguf", "mmproj-F16.gguf"]).projectorFile == "mmproj-F16.gguf")
        }

        @Test func projectorCanBeDisabledOrNamed() throws {
            #expect(try select(gemma, mmproj: .disabled).projectorFile == nil)
            #expect(
                try select(gemma, mmproj: .file("mmproj-model-f16.gguf")).projectorFile == "mmproj-model-f16.gguf"
            )
            #expect(throws: HubError.self) {
                try select(gemma, mmproj: .file("mmproj-missing.gguf"))
            }
            #expect(try select(["model-Q4_K_M.gguf"]).projectorFile == nil)
        }

        @Test func initializerDoesNotResolveFiles() {
            let model = LlamaLanguageModel(modelId: "ggml-org/gemma-3-4b-it-GGUF:Q8_0", mmproj: .disabled)
            #expect(model.modelId == "ggml-org/gemma-3-4b-it-GGUF:Q8_0")
            #expect(model.mmproj == .disabled)
            #expect(model.localModelPath == nil)
            #expect(model.mmprojPath == nil)
            #expect(model.availability == .available)

            let fileModel = LlamaLanguageModel(modelPath: "/path/to/model.gguf", mmprojPath: "/path/to/mmproj.gguf")
            #expect(fileModel.modelId == nil)
            #expect(fileModel.localModelPath == "/path/to/model.gguf")
            #expect(fileModel.mmprojPath == "/path/to/mmproj.gguf")
        }

        /// The deprecated `modelPath` keeps its documented values.
        @available(*, deprecated)
        @Test func deprecatedModelPathReportsEmptyStringBeforeResolution() {
            let model = LlamaLanguageModel(modelId: "ggml-org/gemma-3-4b-it-GGUF")
            #expect(model.modelPath == "")

            let fileModel = LlamaLanguageModel(modelPath: "/path/to/model.gguf")
            #expect(fileModel.modelPath == "/path/to/model.gguf")
        }

        @Test func invalidModelIdThrowsOnFirstRequest() async throws {
            let model = LlamaLanguageModel(modelId: "not-a-repository")
            let session = LanguageModelSession(model: model)
            await #expect(throws: HubError.invalidModelId("not-a-repository")) {
                try await session.respond(to: "Hello")
            }
        }

        @Test func hubClientWithoutCacheThrows() async throws {
            let hub = HubClient(host: HubClient.defaultHost, cache: nil)
            let model = LlamaLanguageModel(modelId: "ggml-org/gemma-3-4b-it-GGUF", hub: hub)
            let session = LanguageModelSession(model: model)
            await #expect(throws: HubError.cacheDisabled) {
                try await session.respond(to: "Hello")
            }
        }
    }

    private let shouldRunLlamaHubTests = {
        // Enable when explicitly requested via environment variable
        if ProcessInfo.processInfo.environment["ENABLE_LLAMA_HUB_TESTS"] != nil {
            return true
        }

        // Skip in CI environments
        if ProcessInfo.processInfo.environment["CI"] != nil {
            return false
        }

        // Skip unless Hugging Face API token is provided
        return ProcessInfo.processInfo.environment["HF_TOKEN"] != nil
    }()

    @Suite("LlamaLanguageModel from the Hub", .enabled(if: shouldRunLlamaHubTests), .serialized)
    struct LlamaHubLanguageModelTests {
        /// Downloads SmolVLM-256M-Instruct's Q8_0 model (175 MB)
        /// and its Q8_0 projector (104 MB) into a temporary cache,
        /// and answers a prompt.
        @Test func downloadsModelAndProjector() async throws {
            let cacheDirectory = FileManager.default.temporaryDirectory
                .appendingPathComponent("LlamaHubLanguageModelTests-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: cacheDirectory) }
            let hub = HubClient(cache: HubCache(cacheDirectory: cacheDirectory))

            let model = LlamaLanguageModel(modelId: "ggml-org/SmolVLM-256M-Instruct-GGUF", hub: hub)
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Say hello.",
                options: GenerationOptions(maximumResponseTokens: 16)
            )
            #expect(!response.content.isEmpty)

            let snapshots = cacheDirectory.appendingPathComponent("models--ggml-org--SmolVLM-256M-Instruct-GGUF")
                .appendingPathComponent("snapshots").path
            let modelPath = try #require(model.localModelPath)
            #expect(modelPath.hasPrefix(snapshots))
            #expect(modelPath.hasSuffix("/SmolVLM-256M-Instruct-Q8_0.gguf"))
            #expect(model.mmprojPath?.hasSuffix("/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf") == true)

            // The blob is named by the file's SHA-256, as llama.cpp names it.
            let blob = try FileManager.default.destinationOfSymbolicLink(atPath: modelPath)
            #expect(blob.hasSuffix("/blobs/2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65"))

            // A new model with an unreachable endpoint uses the cached snapshot.
            let offlineHub = HubClient(
                host: URL(string: "http://127.0.0.1:9")!,
                cache: HubCache(cacheDirectory: cacheDirectory)
            )
            let offline = LlamaLanguageModel(modelId: "ggml-org/SmolVLM-256M-Instruct-GGUF", hub: offlineHub)
            let offlineResponse = try await LanguageModelSession(model: offline).respond(
                to: "Say hello.",
                options: GenerationOptions(maximumResponseTokens: 16)
            )
            #expect(!offlineResponse.content.isEmpty)
            #expect(offline.localModelPath == modelPath)
            #expect(offline.mmprojPath == model.mmprojPath)
        }
    }
#endif  // Llama
