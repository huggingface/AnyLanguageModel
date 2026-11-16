import Foundation

#if Llama
    /// The repository files that a Hub model ID selects.
    struct LlamaHubFileSelection: Equatable, Sendable {
        /// The model file, or every part of a split model in order.
        /// The first element is the file that llama.cpp loads.
        var modelFiles: [String]

        /// The multimodal projector file, if one is selected.
        var projectorFile: String?
    }

    /// Selects GGUF files in a Hugging Face repository
    /// with the rules that llama.cpp 1.0.0 uses for `-hf`.
    ///
    /// The rules come from `common/download.cpp` in ggml-org/llama.cpp at tag `v1.0.0`
    /// (`common_download_split_repo_tag`, `find_best_model`,
    /// `get_split_files`, and `find_best_sibling`).
    /// The resolver works on a list of repository paths
    /// so that it can be tested without the network.
    enum LlamaHubFileResolver {
        /// The quantizations that llama.cpp tries, in order,
        /// when a model ID has no tag.
        static let defaultQuantizations = ["Q4_K_M", "Q8_0"]

        /// Substrings that mark a GGUF file as a sidecar rather than a model.
        /// llama.cpp matches them case-sensitively in the file name.
        static let sidecarKeywords = ["mmproj", "mtp-", "dflash-", "eagle3-", "dspark-"]

        /// Splits a model ID such as `"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"`
        /// into a repository and an optional quantization tag.
        ///
        /// An empty tag, as in `"user/model:"`, is the same as no tag.
        static func parseModelId(_ modelId: String) throws -> (repository: String, tag: String?) {
            let parts = modelId.split(separator: ":", omittingEmptySubsequences: false)
            guard parts.count <= 2 else {
                throw LlamaLanguageModel.HubError.invalidModelId(modelId)
            }
            let repository = String(parts[0])
            let components = repository.split(separator: "/", omittingEmptySubsequences: false)
            guard components.count == 2, components.allSatisfy({ !$0.isEmpty }) else {
                throw LlamaLanguageModel.HubError.invalidModelId(modelId)
            }
            let tag = parts.count == 2 && !parts[1].isEmpty ? String(parts[1]) : nil
            return (repository, tag)
        }

        /// Selects the model files and the projector file from a repository listing.
        ///
        /// - Parameters:
        ///   - files: The repository's file paths, in the order the Hub lists them.
        ///   - repository: The repository ID, used in error messages.
        ///   - tag: The quantization tag from the model ID, if any.
        ///   - file: An explicit model file, which overrides `tag`.
        ///   - mmproj: How to choose the multimodal projector file.
        static func select(
            from files: [String],
            repository: String,
            tag: String?,
            file: String?,
            mmproj: LlamaLanguageModel.Projector
        ) throws -> LlamaHubFileSelection {
            let primary: String
            if let file {
                guard files.contains(file) else {
                    throw LlamaLanguageModel.HubError.fileNotFound(
                        file,
                        repository: repository,
                        available: ggufFiles(in: files)
                    )
                }
                if let split = splitInfo(file), split.count > 1, split.index != 1 {
                    throw LlamaLanguageModel.HubError.notFirstSplitPart(file)
                }
                primary = file
            } else {
                primary = try bestModel(in: files, repository: repository, tag: tag)
            }

            let modelFiles = try splitFiles(for: primary, in: files)

            let projectorFile: String?
            switch mmproj {
            case .disabled:
                projectorFile = nil
            case .automatic:
                projectorFile = bestSibling(of: primary, in: files, keyword: "mmproj")
            case .file(let path):
                guard files.contains(path) else {
                    throw LlamaLanguageModel.HubError.projectorNotFound(
                        path,
                        repository: repository,
                        available: ggufFiles(in: files)
                    )
                }
                projectorFile = path
            }

            return LlamaHubFileSelection(modelFiles: modelFiles, projectorFile: projectorFile)
        }

        // MARK: - Branch selection

        /// Returns `main` if the repository has it, or else the first branch with a valid name,
        /// as llama.cpp's `get_repo_commit` does.
        static func selectBranch(_ names: [String]) -> String? {
            if names.contains("main") {
                return "main"
            }
            return names.first(where: isValidRelativePath)
        }

        /// Returns the commit of the cached `refs/main`,
        /// or else of the first other ref file in name order,
        /// as llama.cpp's `get_cached_ref` does.
        ///
        /// Only files directly in the refs directory count,
        /// and only those whose first line is a 40-character hexadecimal commit.
        /// llama.cpp takes the first other ref in directory order, which isn't defined;
        /// this sorts by name so that the choice doesn't depend on the file system.
        static func cachedCommit(inRefsDirectory directory: URL) -> String? {
            let fileManager = FileManager.default
            guard let names = try? fileManager.contentsOfDirectory(atPath: directory.path) else {
                return nil
            }

            var fallback: String?
            for name in names.sorted() {
                let url = directory.appendingPathComponent(name)
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue,
                    let contents = try? String(contentsOf: url, encoding: .utf8),
                    let line = contents.split(whereSeparator: \.isNewline).first.map(String.init),
                    isCommitHash(line)
                else {
                    continue
                }
                if name == "main" {
                    return line
                }
                if fallback == nil {
                    fallback = line
                }
            }
            return fallback
        }

        /// Whether a string is a 40-character hexadecimal commit hash.
        static func isCommitHash(_ string: String) -> Bool {
            string.count == 40 && string.allSatisfy(\.isHexDigit)
        }

        /// Whether a repository path or branch name stays inside the directory it names a path in.
        ///
        /// It must be relative, with no empty or `..` component,
        /// no backslash, no NUL character, and more than whitespace.
        /// These are the rules of swift-huggingface's cache.
        /// llama.cpp also accepts a `..` that stays inside the directory, such as `a/../b`.
        static func isValidRelativePath(_ path: String) -> Bool {
            guard !path.trimmingCharacters(in: .whitespaces).isEmpty, !path.hasPrefix("/"),
                !path.contains("\\"), !path.contains("\0")
            else {
                return false
            }
            return path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
                !$0.isEmpty && $0 != ".."
            }
        }

        // MARK: - Model selection

        /// Returns the first model file that matches the tag,
        /// or the default quantizations and then any model file when there is no tag.
        private static func bestModel(in files: [String], repository: String, tag: String?) throws -> String {
            let candidates = files.filter { isModelFile($0) && isFirstPart($0) }
            guard !candidates.isEmpty else {
                throw LlamaLanguageModel.HubError.noModelFile(repository: repository)
            }

            for quantization in tag.map({ [$0] }) ?? defaultQuantizations {
                if let match = candidates.first(where: { matchesTag($0, quantization) }) {
                    return match
                }
            }

            // Unlike a missing default, a missing tag doesn't fall back to another file.
            if let tag {
                throw LlamaLanguageModel.HubError.quantizationNotFound(
                    tag,
                    repository: repository,
                    available: ggufFiles(in: files)
                )
            }
            return candidates[0]
        }

        /// Whether a path is a GGUF model rather than an importance matrix or a sidecar.
        static func isModelFile(_ path: String) -> Bool {
            guard path.hasSuffix(".gguf") else { return false }
            let name = fileName(path)
            return !name.contains("imatrix") && !sidecarKeywords.contains(where: name.contains)
        }

        /// Whether a path matches a tag as a whole token, ignoring case.
        ///
        /// The tag must follow the start of the path or `-`, `.`, or `/`,
        /// and be followed by `.` or `-`, so that `F16` doesn't match `BF16`.
        /// llama.cpp builds this regular expression from the tag without escaping it;
        /// this resolver matches the tag literally.
        static func matchesTag(_ path: String, _ tag: String) -> Bool {
            let pattern = "(^|[-./])" + NSRegularExpression.escapedPattern(for: tag) + "[.-]"
            guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
                return false
            }
            return regex.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)) != nil
        }

        // MARK: - Split files

        /// The parts of a file name such as `Model-Q4_K_M-00001-of-00003.gguf`.
        struct SplitInfo: Equatable {
            /// The path without `.gguf` and without the split suffix.
            var prefix: String
            /// The upper-cased quantization tag at the end of the prefix, or an empty string.
            var tag: String
            /// The one-based part number, or 1 for a file that isn't split.
            var index: Int
            /// The number of parts, or 1 for a file that isn't split.
            var count: Int
        }

        /// Returns the split information for a GGUF path, or `nil` for another file.
        static func splitInfo(_ path: String) -> SplitInfo? {
            guard path.hasSuffix(".gguf") else { return nil }
            var prefix = String(path.dropLast(".gguf".count))
            var index = 1
            var count = 1

            if let match = firstMatch(of: #"^(.+)-([0-9]{5})-of-([0-9]{5})$"#, in: prefix),
                let base = match[1], let part = match[2].flatMap(Int.init), let total = match[3].flatMap(Int.init)
            {
                prefix = base
                index = part
                count = total
            }

            let tag = firstMatch(of: #"[-.]([A-Z0-9_]+)$"#, in: prefix)?[1]?.uppercased() ?? ""
            return SplitInfo(prefix: prefix, tag: tag, index: index, count: count)
        }

        private static func isFirstPart(_ path: String) -> Bool {
            guard let split = splitInfo(path) else { return false }
            return split.count <= 1 || split.index == 1
        }

        /// Returns every part of a split model in order, or the file itself.
        ///
        /// llama.cpp loads the first part and finds the others next to it by name,
        /// so a missing part is an error here rather than at load time.
        private static func splitFiles(for primary: String, in files: [String]) throws -> [String] {
            guard let split = splitInfo(primary), split.count > 1 else {
                return [primary]
            }

            var parts: [Int: String] = [:]
            for path in files {
                guard let info = splitInfo(path), info.count == split.count, info.prefix == split.prefix else {
                    continue
                }
                parts[info.index] = path
            }

            let missing = (1 ... split.count).filter { parts[$0] == nil }.map {
                split.prefix + "-" + zeroPadded($0) + "-of-" + zeroPadded(split.count) + ".gguf"
            }
            guard missing.isEmpty else {
                throw LlamaLanguageModel.HubError.incompleteSplit(primary, missing: missing)
            }
            return (1 ... split.count).map { parts[$0]! }
        }

        // MARK: - Projector selection

        /// Returns the bits of a quantization tag:
        /// the first run of digits, so `Q4_K_M` is 4, `F16` and `BF16` are 16,
        /// and a tag without digits is 0.
        static func quantizationBits(_ path: String) -> Int {
            let tag = splitInfo(path)?.tag ?? ""
            let digits = tag.drop(while: { !$0.isNumber }).prefix(while: \.isNumber)
            return Int(digits) ?? 0
        }

        /// Returns the GGUF file containing `keyword` that best pairs with `model`.
        ///
        /// The file must be in the model's directory or one of its ancestors.
        /// The deepest directory wins,
        /// then the smallest difference in quantization bits from the model,
        /// then the first file in listing order.
        static func bestSibling(of model: String, in files: [String], keyword: String) -> String? {
            let modelDirectory = directoryComponents(model)
            let modelBits = quantizationBits(model)

            var best: (path: String, depth: Int, difference: Int)?
            for path in files where path.hasSuffix(".gguf") && path.contains(keyword) {
                let directory = directoryComponents(path)
                guard directory.count <= modelDirectory.count,
                    Array(modelDirectory.prefix(directory.count)) == directory
                else {
                    continue
                }
                let difference = abs(quantizationBits(path) - modelBits)
                if let current = best,
                    directory.count < current.depth
                        || (directory.count == current.depth && difference >= current.difference)
                {
                    continue
                }
                best = (path, directory.count, difference)
            }
            return best?.path
        }

        // MARK: - Helpers

        private static func fileName(_ path: String) -> Substring {
            path.split(separator: "/", omittingEmptySubsequences: false).last ?? Substring(path)
        }

        private static func directoryComponents(_ path: String) -> [Substring] {
            Array(path.split(separator: "/", omittingEmptySubsequences: false).dropLast())
        }

        private static func ggufFiles(in files: [String]) -> [String] {
            files.filter { $0.hasSuffix(".gguf") }
        }

        private static func zeroPadded(_ value: Int) -> String {
            let digits = String(value)
            return String(repeating: "0", count: max(0, 5 - digits.count)) + digits
        }

        /// Returns the capture groups of the first case-insensitive match,
        /// with the whole match at index 0.
        private static func firstMatch(of pattern: String, in string: String) -> [String?]? {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
                let match = regex.firstMatch(in: string, range: NSRange(string.startIndex..., in: string))
            else {
                return nil
            }
            return (0 ..< match.numberOfRanges).map { group in
                Range(match.range(at: group), in: string).map { String(string[$0]) }
            }
        }
    }
#endif  // Llama
