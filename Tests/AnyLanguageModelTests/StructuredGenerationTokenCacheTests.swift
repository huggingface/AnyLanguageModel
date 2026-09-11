import Testing

@testable import AnyLanguageModel

struct StructuredGenerationTokenCacheTests {
    /// Immutable vocabulary and synchronized observations, safe to share across tasks.
    private struct VocabularyBackend: TokenBackend, Sendable {
        struct Observations: Sendable {
            var textCalls: [Int: Int] = [:]
            var specialCalls: [Int: Int] = [:]
            var masks: [Set<Int>] = []
        }

        let texts: [Int: String]
        let specialTokens: Set<Int>
        let vocabSize: Int
        let observations = Locked(Observations())
        var endTokens: Set<Int> = [63]
        var eosToken: Int { 63 }
        var remainingTokens = 64
        var totalTokenBudget: Int { 64 }
        var queue: [Int] = []

        init(texts: [Int: String], specialTokens: Set<Int> = [], vocabSize: Int = 64) {
            self.texts = texts
            self.specialTokens = specialTokens
            self.vocabSize = vocabSize
        }

        func tokenize(_ text: String) throws -> [Int] {
            texts.first { $0.value == text }.map { [$0.key] } ?? []
        }

        func tokenText(_ token: Int) -> String? {
            observations.withLock { $0.textCalls[token, default: 0] += 1 }
            return texts[token]
        }

        func isSpecialToken(_ token: Int) -> Bool {
            observations.withLock { $0.specialCalls[token, default: 0] += 1 }
            return specialTokens.contains(token)
        }

        mutating func decode(_ token: Int) async throws {
            remainingTokens -= 1
        }

        mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
            observations.withLock { $0.masks.append(allowedTokens) }
            guard !queue.isEmpty else { throw ConstrainedGenerationError.tokenizationFailed }
            let token = queue.removeFirst()
            guard allowedTokens.contains(token) else { throw ConstrainedGenerationError.tokenizationFailed }
            return token
        }
    }

    private var baseTexts: [Int: String] {
        var texts = [0: "\"", 1: ",", 2: "}", 3: "]", 4: ":", 15: "-", 16: ".", 63: "<eos>"]
        for digit in 0 ... 9 { texts[5 + digit] = String(digit) }
        return texts
    }

    private func expectEqual(_ lhs: StructuredGenerationTokenSets, _ rhs: StructuredGenerationTokenSets) {
        #expect(lhs.stringContent == rhs.stringContent)
        #expect(lhs.integerContent == rhs.integerContent)
        #expect(lhs.decimalContent == rhs.decimalContent)
    }

    @Test func coldConstructionClassifiesAllSetsInOnePass() {
        let backend = VocabularyBackend(
            texts: [0: "word", 1: "42", 2: "-", 3: ".", 4: "-1.5", 5: "", 6: "9", 7: "\""],
            specialTokens: [6],
            vocabSize: 9
        )
        let tokens = StructuredGenerationTokenCache().tokens(for: backend)
        #expect(tokens.stringContent == [0, 1, 2, 3, 4])
        #expect(tokens.integerContent == [1, 2])
        #expect(tokens.decimalContent == [1, 2, 3, 4])
        let observations = backend.observations.withLock { $0 }
        #expect(observations.specialCalls == Dictionary(uniqueKeysWithValues: (0 ..< 9).map { ($0, 1) }))
        #expect(observations.textCalls == [0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 7: 1, 8: 1])
    }

    private struct ClassificationCase: Sendable {
        let text: String?
        var special = false
        var string = false
        var integer = false
        var decimal = false
    }

    @Test(arguments: [
        ClassificationCase(text: nil),
        ClassificationCase(text: ""),
        ClassificationCase(text: "42", special: true),
        ClassificationCase(text: "42", string: true, integer: true, decimal: true),
        ClassificationCase(text: "-12", string: true, integer: true, decimal: true),
        ClassificationCase(text: "-", string: true, integer: true, decimal: true),
        ClassificationCase(text: ".", string: true, decimal: true),
        ClassificationCase(text: "12.5", string: true, decimal: true),
        ClassificationCase(text: "--", string: true),
        ClassificationCase(text: "-.", string: true),
        ClassificationCase(text: "1-2", string: true, integer: true, decimal: true),
        ClassificationCase(text: "1..2", string: true, decimal: true),
        ClassificationCase(text: "１２", string: true),
        ClassificationCase(text: "²", string: true),
        ClassificationCase(text: "١", string: true),
        ClassificationCase(text: "1e2", string: true),
        ClassificationCase(text: "+1", string: true),
        ClassificationCase(text: " ", string: true),
        ClassificationCase(text: "  "),
        ClassificationCase(text: "\t"),
        ClassificationCase(text: "\n"),
        ClassificationCase(text: "\r"),
        ClassificationCase(text: "\u{00A0}"),
        ClassificationCase(text: " a ", string: true),
        ClassificationCase(text: "a\nb"),
        ClassificationCase(text: "\""),
        ClassificationCase(text: "“"),
        ClassificationCase(text: "’"),
        ClassificationCase(text: "\\"),
        ClassificationCase(text: "\\n"),
        ClassificationCase(text: "】"),
        ClassificationCase(text: "hello", string: true),
        ClassificationCase(text: "é", string: true),
        ClassificationCase(text: "😀", string: true),
    ])
    private func classifierPreservesExistingPolicy(testCase: ClassificationCase) {
        let backend = VocabularyBackend(
            texts: testCase.text.map { [0: $0] } ?? [:],
            specialTokens: testCase.special ? [0] : [],
            vocabSize: 1
        )
        let tokens = StructuredGenerationTokenCache().tokens(for: backend)
        #expect(tokens.stringContent == (testCase.string ? [0] : []))
        #expect(tokens.integerContent == (testCase.integer ? [0] : []))
        #expect(tokens.decimalContent == (testCase.decimal ? [0] : []))
    }

    @Test func noCacheBuildsFreshSetsAndGeneratesCorrectOutput() async throws {
        var backend = VocabularyBackend(texts: baseTexts)
        backend.queue = [6, 2]
        for pass in 1 ... 2 {
            var generator = try ConstrainedJSONGenerator(backend: backend, schema: Int.generationSchema)
            #expect(backend.observations.withLock { $0.specialCalls.values.reduce(0, +) } == pass * 64)
            // Each preceding generation reads the emitted numeric token once.
            #expect(backend.observations.withLock { $0.textCalls.values.reduce(0, +) } == pass * 64 + pass - 1)
            #expect(try await generator.generate() == "1")
        }
    }

    @Test(arguments: [false, true])
    func unsampledNumericDifferenceIsIndependent(reverse: Bool) async throws {
        // Token 20 is absent from the former endpoint, midpoint, EOS, end-token,
        // and individual digit/sign/point samples. All those mappings agree.
        for numeric in (reverse ? [true, false] : [false, true]) {
            var texts = baseTexts
            texts[20] = numeric ? "42" : "word"
            var backend = VocabularyBackend(texts: texts)
            backend.queue = numeric ? [20, 2] : [6, 2]
            let cache = StructuredGenerationTokenCache()
            let tokens = cache.tokens(for: backend)
            #expect(tokens.integerContent == Set(5 ... 15).union(numeric ? [20] : []))
            #expect(tokens.decimalContent == Set(5 ... 16).union(numeric ? [20] : []))
            for schema in [Int.generationSchema, Double.generationSchema] {
                var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema, tokenCache: cache)
                #expect(try await generator.generate() == (numeric ? "42" : "1"))
            }
            #expect(backend.observations.withLock { $0.masks.allSatisfy { $0.contains(20) == numeric } })
        }
    }

    @Test(arguments: [false, true])
    func unsampledStringDifferenceIsIndependent(reverse: Bool) async throws {
        for valid in (reverse ? [true, false] : [false, true]) {
            var texts = baseTexts
            texts[20] = valid ? "word" : "\\"
            var backend = VocabularyBackend(texts: texts, specialTokens: [63])
            backend.queue = [6, 0]
            let cache = StructuredGenerationTokenCache()
            let tokens = cache.tokens(for: backend)
            #expect(tokens.stringContent == Set(1 ... 16).union(valid ? [20] : []))
            var generator = try ConstrainedJSONGenerator(
                backend: backend,
                schema: String.generationSchema,
                tokenCache: cache
            )
            #expect(try await generator.generate() == "\"1\"")
            #expect(backend.observations.withLock { $0.masks.allSatisfy { $0.contains(20) == valid } })
        }
    }

    @Test(arguments: [false, true])
    func specialTokenClassificationBelongsToLoadedVocabulary(reverse: Bool) {
        for special in (reverse ? [true, false] : [false, true]) {
            let backend = VocabularyBackend(texts: [20: "42"], specialTokens: special ? [20] : [])
            let tokens = StructuredGenerationTokenCache().tokens(for: backend)
            #expect(tokens.stringContent == (special ? [] : [20]))
            #expect(tokens.integerContent == (special ? [] : [20]))
            #expect(tokens.decimalContent == (special ? [] : [20]))
        }
    }

    @Test func endTokenChangesReuseContentAndApplyCurrentMasks() async throws {
        var texts = baseTexts
        texts[20] = "word"
        let original = VocabularyBackend(texts: texts, specialTokens: [63])
        let cache = StructuredGenerationTokenCache()
        for ends: Set<Int> in [[20, 6], [63]] {
            var backend = original
            backend.endTokens = ends
            backend.queue = [7, 0]
            var generator = try ConstrainedJSONGenerator(
                backend: backend,
                schema: String.generationSchema,
                tokenCache: cache
            )
            #expect(try await generator.generate() == "\"2\"")
            let masks = backend.observations.withLock { Array($0.masks.suffix(2)) }
            let expectedInitial = Set(1 ... 16).union([20]).subtracting(ends)
            #expect(masks[0] == expectedInitial)
            #expect(masks[1] == expectedInitial.union(ends).union([0]))

            for schema in [Int.generationSchema, Double.generationSchema] {
                backend.queue = [7, 2]
                var number = try ConstrainedJSONGenerator(backend: backend, schema: schema, tokenCache: cache)
                #expect(try await number.generate() == "2")
                // Numeric masks preserve end tokens if their content is numeric.
                #expect(backend.observations.withLock { $0.masks.suffix(2).allSatisfy { $0.contains(6) } })
            }
        }
        #expect(original.observations.withLock { $0.specialCalls.values.reduce(0, +) } == 64)
        #expect(original.observations.withLock { $0.textCalls.values.reduce(0, +) } == 69)
    }

    @Test func vocabularySizesHaveSeparateEntries() {
        let cache = StructuredGenerationTokenCache()
        let small = VocabularyBackend(texts: [0: "42", 2: "1.5"], vocabSize: 2)
        let large = VocabularyBackend(texts: small.texts, vocabSize: 3)
        let first = cache.tokens(for: small)
        #expect(first.stringContent == [0])
        #expect(first.integerContent == [0])
        #expect(first.decimalContent == [0])
        let second = cache.tokens(for: large)
        #expect(second.stringContent == [0, 2])
        #expect(second.integerContent == [0])
        #expect(second.decimalContent == [0, 2])
        expectEqual(cache.tokens(for: small), first)
        #expect(small.observations.withLock { $0.specialCalls == [0: 1, 1: 1] })
        #expect(large.observations.withLock { $0.specialCalls == [0: 1, 1: 1, 2: 1] })
    }

    @Test func concurrentFirstUsePerformsExactlyOneScan() async {
        let backend = VocabularyBackend(texts: [20: "42", 21: "."], specialTokens: [63])
        let cache = StructuredGenerationTokenCache()
        await withTaskGroup(of: StructuredGenerationTokenSets.self) { group in
            for _ in 0 ..< 32 {
                group.addTask { cache.tokens(for: backend) }
            }
            for await tokens in group {
                #expect(tokens.stringContent == [20, 21])
                #expect(tokens.integerContent == [20])
                #expect(tokens.decimalContent == [20, 21])
            }
        }
        #expect(
            backend.observations.withLock { $0.specialCalls }
                == Dictionary(
                    uniqueKeysWithValues: (0 ..< 64).map { ($0, 1) }
                )
        )
        #expect(
            backend.observations.withLock { $0.textCalls }
                == Dictionary(
                    uniqueKeysWithValues: (0 ..< 63).map { ($0, 1) }
                )
        )
    }

    @Test func reloadCreatesFreshCacheAndActiveOwnershipControlsRelease() throws {
        struct LoadedOwner {
            let label = "same-model"
            let backend: VocabularyBackend
            let cache = StructuredGenerationTokenCache()
        }
        var loaded: LoadedOwner? = LoadedOwner(backend: VocabularyBackend(texts: [20: "42"]))
        var activeRequest = loaded
        weak var oldCache: StructuredGenerationTokenCache?
        oldCache = loaded?.cache
        do {
            let initial = try #require(loaded)
            #expect(initial.cache.tokens(for: initial.backend).integerContent == [20])
            loaded = LoadedOwner(backend: VocabularyBackend(texts: [20: "word"]))
            let reloaded = try #require(loaded)
            #expect(reloaded.label == initial.label)
            #expect(reloaded.cache !== initial.cache)
            #expect(reloaded.cache.tokens(for: reloaded.backend).integerContent.isEmpty)
        }
        #expect(oldCache != nil)
        #expect(activeRequest?.cache === oldCache)
        do {
            let active = try #require(activeRequest)
            #expect(active.cache.tokens(for: active.backend).integerContent == [20])
            #expect(active.backend.observations.withLock { $0.specialCalls.values.reduce(0, +) } == 64)
        }
        activeRequest = nil
        #expect(oldCache == nil)
        withExtendedLifetime(loaded) {}
    }

    @Test func cacheDoesNotRetainBackendOrOutliveOwner() {
        weak var releasedCache: StructuredGenerationTokenCache?
        weak var releasedObservations: Locked<VocabularyBackend.Observations>?
        var activeCache: StructuredGenerationTokenCache?
        do {
            let backend = VocabularyBackend(texts: [20: "42"])
            let cache = StructuredGenerationTokenCache()
            releasedCache = cache
            releasedObservations = backend.observations
            _ = cache.tokens(for: backend)
            activeCache = cache
        }
        #expect(releasedObservations == nil)
        #expect(releasedCache != nil)
        withExtendedLifetime(activeCache) {}
        activeCache = nil
        #expect(releasedCache == nil)
    }

    @Test func structuralValidationPrecedesScanning() {
        let backend = VocabularyBackend(texts: [0: "\"", 1: ","])
        #expect(throws: ConstrainedGenerationError.self) {
            _ = try ConstrainedJSONGenerator(
                backend: backend,
                schema: String.generationSchema,
                tokenCache: StructuredGenerationTokenCache()
            )
        }
        #expect(backend.observations.withLock { $0.textCalls.isEmpty && $0.specialCalls.isEmpty })
    }
}
