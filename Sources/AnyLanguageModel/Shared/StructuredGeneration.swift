import Foundation

// MARK: - Token Backend

/// A token-level backend used by ``ConstrainedJSONGenerator``.
///
/// Implementations provide tokenization, sampling, and decoding state so the
/// generator can constrain output to valid JSON for a schema.
protocol TokenBackend {
    func tokenize(_ text: String) throws -> [Int]
    func tokenText(_ token: Int) -> String?
    func isSpecialToken(_ token: Int) -> Bool
    mutating func decode(_ token: Int) async throws
    mutating func sample(from allowedTokens: Set<Int>) async throws -> Int

    var eosToken: Int { get }
    var endTokens: Set<Int> { get }
    var vocabSize: Int { get }
    var remainingTokens: Int { get set }
    var totalTokenBudget: Int { get }
}

/// Heuristics for deciding when to include optional properties in generated output.
private enum OptionalPropertyBudget {
    /// Minimum absolute number of tokens that should remain before we consider
    /// adding optional properties.
    static let minimumRemainingTokens = 8

    /// Require at least this fraction of the total budget (divisor form).
    static let minimumBudgetDivisor = 10

    static func minimumBudget(totalTokenBudget: Int) -> Int {
        let proportionalBudget = totalTokenBudget / minimumBudgetDivisor
        return max(minimumRemainingTokens, proportionalBudget)
    }
}

private final class StringTokenCache: @unchecked Sendable {
    static let shared = StringTokenCache()

    struct Key: Hashable {
        let vocabSize: Int
        let eosToken: Int
        let endTokens: Set<Int>
        let sampleTexts: [String]
    }

    private var cache: [Key: Set<Int>] = [:]
    private let lock = NSLock()

    func tokens(for key: Key) -> Set<Int>? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]
    }

    func store(_ tokens: Set<Int>, for key: Key) {
        lock.lock()
        cache[key] = tokens
        lock.unlock()
    }
}

// MARK: - JSON Generator

/// Generates JSON that conforms to a schema using constrained token sampling.
struct ConstrainedJSONGenerator<Backend: TokenBackend> {
    /// Minimum per-string token allocation to ensure small budgets still yield content.
    private static var freeStringMinTokenLimit: Int { 16 }
    /// Divisor used to compute the per-string share of the total token budget.
    private static var freeStringTokenBudgetDivisor: Int { 16 }

    /// Upper bounds for numeric token generation heuristics.
    private static var maxIntegerTokenLimit: Int { 20 }
    private static var maxDecimalTokenLimit: Int { 32 }

    /// Heuristics for default array sizes when no bounds are specified.
    private static var arrayDefaultCountDivisor: Int { 32 }
    private static var arrayDefaultCountMax: Int { 16 }

    private var backend: Backend
    private let schema: GenerationSchema
    private var emittedText = ""
    private let quoteToken: Int

    private let stringTerminators: Set<Int>
    private let stringInitialAllowedTokens: Set<Int>
    private let stringContinuationAllowedTokens: Set<Int>

    private let basicTerminators: Set<Int>
    private let integerTerminators: Set<Int>
    private let doubleTerminators: Set<Int>

    /// Creates a constrained JSON generator.
    ///
    /// - Parameters:
    ///   - backend: A backend that provides tokenization and sampling.
    ///   - schema: The generation schema to satisfy.
    /// - Throws: ``ConstrainedGenerationError`` when required tokens cannot be tokenized.
    init(backend: Backend, schema: GenerationSchema) throws {
        self.backend = backend
        self.schema = schema

        let quoteToken = try Self.singleToken(for: "\"", backend: backend)
        self.quoteToken = quoteToken

        self.stringTerminators = backend.endTokens.union([quoteToken])

        var structuralTerminators = Set<Int>()
        for structuralText in [",", "}", "]", ":"] {
            let token = try Self.singleToken(for: structuralText, backend: backend)
            structuralTerminators.insert(token)
        }
        self.basicTerminators = structuralTerminators
        self.integerTerminators = Self.buildValidIntegerTokens(backend: backend).union(structuralTerminators)
        self.doubleTerminators = Self.buildValidDecimalTokens(backend: backend).union(structuralTerminators)

        let stringContentTokens = Self.buildValidStringTokens(backend: backend)
        self.stringInitialAllowedTokens = stringContentTokens
        self.stringContinuationAllowedTokens = stringContentTokens.union(stringTerminators)
    }

    /// Generates a JSON string that conforms to the schema.
    ///
    /// - Returns: A JSON string that satisfies the schema. If the backend emits
    ///   an end token early, the partial output is returned.
    /// - Throws: ``ConstrainedGenerationError`` if generation fails.
    mutating func generate() async throws -> String {
        do {
            return try await generateNode(schema.root)
        } catch let error as ConstrainedGenerationError {
            if case .earlyTermination(let partial) = error {
                return partial
            }
            throw error
        }
    }

    private static func singleToken(for text: String, backend: Backend) throws -> Int {
        let tokens = try backend.tokenize(text)
        guard tokens.count == 1, let token = tokens.first else {
            throw ConstrainedGenerationError.unsupportedTokenizer("Expected single-token encoding for '\(text)'")
        }
        return token
    }

    private static func buildValidStringTokens(backend: Backend) -> Set<Int> {
        let cacheKey = stringTokenCacheKey(for: backend)
        if let cached = StringTokenCache.shared.tokens(for: cacheKey) {
            return cached
        }

        let allowedWhitespace: Set<Character> = [" ", "\t", "\n"]
        var allowed = Set<Int>()
        allowed.reserveCapacity(backend.vocabSize / 4)

        for token in 0 ..< backend.vocabSize {
            if backend.endTokens.contains(token) { continue }
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            guard text.allSatisfy({ $0.isValidJSONStringCharacter }) else { continue }

            if text.allSatisfy({ $0.isWhitespace }) {
                if text.count == 1, let char = text.first, allowedWhitespace.contains(char) {
                    allowed.insert(token)
                }
            } else {
                allowed.insert(token)
            }
        }

        StringTokenCache.shared.store(allowed, for: cacheKey)
        return allowed
    }

    private static func stringTokenCacheKey(for backend: Backend) -> StringTokenCache.Key {
        let sampleTokenIds = sampleTokenIds(for: backend)
        let sampleTexts = sampleTokenIds.map { backend.tokenText($0) ?? "" }
        return StringTokenCache.Key(
            vocabSize: backend.vocabSize,
            eosToken: backend.eosToken,
            endTokens: backend.endTokens,
            sampleTexts: sampleTexts
        )
    }

    private static func sampleTokenIds(for backend: Backend) -> [Int] {
        let vocabSize = max(0, backend.vocabSize)
        var samples: Set<Int> = [
            0,
            1,
            2,
            max(0, vocabSize / 2),
            max(0, vocabSize - 1),
            backend.eosToken,
        ]
        samples.formUnion(backend.endTokens)
        return samples.filter { $0 >= 0 && $0 < vocabSize }.sorted()
    }

    private static func buildValidIntegerTokens(backend: Backend) -> Set<Int> {
        var allowed = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            if text.allSatisfy({ $0.isNumber || $0 == "-" }),
                text.contains(where: { $0.isNumber })
            {
                allowed.insert(token)
            }
        }
        return allowed
    }

    private static func buildValidDecimalTokens(backend: Backend) -> Set<Int> {
        var allowed = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            if text.allSatisfy({ $0.isNumber || $0 == "-" || $0 == "." }),
                text.contains(where: { $0.isNumber })
            {
                allowed.insert(token)
            }
        }
        return allowed
    }

    private mutating func emit(_ text: String) async throws -> String {
        for token in try backend.tokenize(text) {
            guard backend.remainingTokens > 0 else {
                throw ConstrainedGenerationError.tokenBudgetExceeded
            }
            try await backend.decode(token)
        }
        emittedText += text
        return text
    }

    private func maxFreeStringTokens() -> Int {
        let perStringLimit = max(
            Self.freeStringMinTokenLimit,
            backend.totalTokenBudget / Self.freeStringTokenBudgetDivisor
        )
        let remainingAfterClosingQuote = max(0, backend.remainingTokens - 1)
        return min(remainingAfterClosingQuote, perStringLimit)
    }

    private mutating func generateFreeString(maxTokens: Int) async throws -> String {
        var result = ""
        var generated = 0

        while backend.remainingTokens > 0, generated < maxTokens {
            let allowed = result.isEmpty ? stringInitialAllowedTokens : stringContinuationAllowedTokens
            let token = try await backend.sample(from: allowed)
            if backend.endTokens.contains(token) {
                throw ConstrainedGenerationError.earlyTermination(emittedText)
            }
            if token == quoteToken { break }

            let text = backend.tokenText(token) ?? ""
            result += text
            emittedText += text
            generated += 1
            try await backend.decode(token)
        }

        return result
    }

    private mutating func generateChoice(_ candidates: [String]) async throws -> String {
        guard !candidates.isEmpty else {
            throw ConstrainedGenerationError.tokenizationFailed
        }

        let tokenized = try candidates.map { try backend.tokenize($0) }
        for (candidate, tokens) in zip(candidates, tokenized) {
            if candidate.isEmpty { continue }
            if tokens.isEmpty {
                throw ConstrainedGenerationError.tokenizationFailed
            }
        }

        let hasEmptyCandidate = candidates.contains("")
        let hasPrefixCollision = Self.hasPrefixCollision(tokenized: tokenized)
        if hasEmptyCandidate || hasPrefixCollision {
            let chosen = deterministicChoice(from: candidates)
            if !chosen.isEmpty {
                _ = try await emit(chosen)
            }
            return chosen
        }

        var prefixes = tokenized
        var emitted = ""
        var position = 0

        while backend.remainingTokens > 0 {
            if prefixes.contains(where: { $0.count == position }) { break }

            let allowed = Set(
                prefixes.compactMap { tokens -> Int? in
                    guard position < tokens.count else { return nil }
                    return tokens[position]
                }
            )

            let token = try await backend.sample(from: allowed)
            if backend.endTokens.contains(token) {
                throw ConstrainedGenerationError.earlyTermination(emittedText)
            }
            let text = backend.tokenText(token) ?? ""
            emitted += text
            emittedText += text
            try await backend.decode(token)

            prefixes = prefixes.filter { $0.count > position && $0[position] == token }
            position += 1
            if prefixes.isEmpty { break }
        }

        return emitted
    }

    private func maxNumberTokens(for node: GenerationSchema.NumberNode) -> Int {
        var limit =
            node.integerOnly
            ? Self.maxIntegerTokenLimit
            : Self.maxDecimalTokenLimit

        if node.integerOnly, let minimum = node.minimum, let maximum = node.maximum {
            let maxAbs = max(abs(minimum), abs(maximum))
            if maxAbs.isFinite, maxAbs <= Double(Int64.max) {
                let digits = max(1, String(Int64(maxAbs.rounded(.down))).count)
                limit = max(limit, digits + 1)
            }
        }

        return min(backend.remainingTokens, limit)
    }

    private mutating func generateNumber(_ node: GenerationSchema.NumberNode) async throws -> String {
        let allowedTokens = node.integerOnly ? integerTerminators : doubleTerminators
        let numericTokens = allowedTokens.subtracting(basicTerminators)
        var result = ""
        let maxTokens = maxNumberTokens(for: node)
        var generatedTokens = 0

        while backend.remainingTokens > 0, generatedTokens < maxTokens {
            let candidates = result.isEmpty ? numericTokens : allowedTokens
            guard !candidates.isEmpty else {
                throw ConstrainedGenerationError.tokenizationFailed
            }
            let token = try await backend.sample(from: candidates)
            if backend.endTokens.contains(token) {
                throw ConstrainedGenerationError.earlyTermination(emittedText)
            }
            if basicTerminators.contains(token) { break }

            guard let text = backend.tokenText(token) else { break }
            result += text
            emittedText += text
            generatedTokens += 1
            try await backend.decode(token)
        }

        guard !result.isEmpty else {
            throw ConstrainedGenerationError.numberOutOfRange("Missing number value")
        }
        return try validateNumberString(result, node: node)
    }

    private func validateNumberString(_ text: String, node: GenerationSchema.NumberNode) throws -> String {
        if node.integerOnly {
            guard let value = Int(text) else {
                throw ConstrainedGenerationError.numberOutOfRange("Invalid integer: \(text)")
            }
            if let minimum = node.minimum, Double(value) < minimum {
                throw ConstrainedGenerationError.numberOutOfRange("Integer \(value) is below minimum \(minimum)")
            }
            if let maximum = node.maximum, Double(value) > maximum {
                throw ConstrainedGenerationError.numberOutOfRange("Integer \(value) exceeds maximum \(maximum)")
            }
            return text
        } else {
            guard let value = Double(text), !value.isNaN, value.isFinite else {
                throw ConstrainedGenerationError.numberOutOfRange("Invalid number: \(text)")
            }
            if let minimum = node.minimum, value < minimum {
                throw ConstrainedGenerationError.numberOutOfRange("Number \(value) is below minimum \(minimum)")
            }
            if let maximum = node.maximum, value > maximum {
                throw ConstrainedGenerationError.numberOutOfRange("Number \(value) exceeds maximum \(maximum)")
            }
            return text
        }
    }

    private mutating func generateNode(_ node: GenerationSchema.Node) async throws -> String {
        guard backend.remainingTokens > 0 else {
            throw ConstrainedGenerationError.tokenBudgetExceeded
        }

        switch node {
        case .object(let objectNode):
            return try await generateObject(objectNode)
        case .array(let arrayNode):
            return try await generateArray(arrayNode)
        case .string(let stringNode):
            return try await generateString(stringNode)
        case .number(let numberNode):
            return try await generateNumber(numberNode)
        case .boolean:
            return try await generateChoice(["true", "false"])
        case .ref(let typeName):
            guard let referenced = schema.defs[typeName] else {
                throw ConstrainedGenerationError.missingReference(typeName)
            }
            return try await generateNode(referenced)
        case .anyOf(let variants):
            guard !variants.isEmpty else {
                throw ConstrainedGenerationError.emptyAnyOf
            }
            if variants.count == 1 {
                return try await generateNode(variants[0])
            }
            // Choose the first variant to keep selection deterministic.
            return try await generateNode(variants[0])
        }
    }

    private mutating func generateObject(_ node: GenerationSchema.ObjectNode) async throws -> String {
        let keys = node.properties.keys.sorted()
        let includedKeys = keys.filter { shouldIncludeOptionalProperty($0, required: node.required) }
        var output = try await emit("{")

        for (index, key) in includedKeys.enumerated() {
            output += try await emit("\"\(key)\":")
            output += try await generateNode(node.properties[key] ?? .string(.init()))

            if index < includedKeys.count - 1 {
                output += try await emit(",")
            }
        }

        output += try await emit("}")
        return output
    }

    private mutating func generateArray(_ node: GenerationSchema.ArrayNode) async throws -> String {
        // Derive a default item count from the total token budget when the schema
        // does not specify explicit minItems/maxItems. We use a small fraction of the
        // budget and clamp it to a reasonable range to avoid overlong arrays.
        let budgetBasedCount = backend.totalTokenBudget / Self.arrayDefaultCountDivisor
        let defaultCount = max(1, min(Self.arrayDefaultCountMax, budgetBasedCount))
        let count: Int

        if let minItems = node.minItems, let maxItems = node.maxItems {
            if minItems > maxItems {
                throw ConstrainedGenerationError.invalidArrayBounds(
                    "Minimum items \(minItems) exceeds maximum \(maxItems)"
                )
            }
            let rangeSize = maxItems - minItems + 1
            let offset = rangeSize > 0 ? backend.totalTokenBudget % rangeSize : 0
            count = minItems + offset
        } else if let minItems = node.minItems {
            count = minItems
        } else if let maxItems = node.maxItems {
            count = maxItems
        } else {
            count = defaultCount
        }
        var output = try await emit("[")

        for index in 0 ..< count {
            output += try await generateNode(node.items)
            if index < count - 1 {
                output += try await emit(",")
            }
        }

        output += try await emit("]")
        return output
    }

    private mutating func generateString(_ node: GenerationSchema.StringNode) async throws -> String {
        var output = try await emit("\"")
        let content: String
        let pattern = node.pattern
        let regex = try pattern.map { try compilePattern($0) }

        if let choices = node.enumChoices, !choices.isEmpty {
            let applicableChoices: [String]
            if let pattern, let regex {
                let filtered = choices.filter { matchesPattern($0, regex: regex) }
                guard !filtered.isEmpty else {
                    throw ConstrainedGenerationError.patternMismatch(
                        "No enum choices match pattern '\(pattern)'"
                    )
                }
                applicableChoices = filtered
            } else {
                applicableChoices = choices
            }
            content = try await generateChoice(applicableChoices)
        } else {
            content = try await generateFreeString(maxTokens: maxFreeStringTokens())
        }

        if let pattern, let regex {
            if !matchesPattern(content, regex: regex) {
                throw ConstrainedGenerationError.patternMismatch(
                    "Value '\(content)' does not match pattern '\(pattern)'"
                )
            }
        }

        output += content
        output += try await emit("\"")
        return output
    }

    private func shouldIncludeOptionalProperty(_ key: String, required: Set<String>) -> Bool {
        if required.contains(key) { return true }
        let minimumBudget = OptionalPropertyBudget.minimumBudget(totalTokenBudget: backend.totalTokenBudget)
        guard backend.remainingTokens > minimumBudget else { return false }
        let hash = key.utf8.reduce(0) { ($0 &* 31) &+ Int($1) }
        let combined = hash ^ backend.totalTokenBudget
        return combined % 2 == 0
    }

    private func deterministicChoice(from candidates: [String]) -> String {
        guard !candidates.isEmpty else { return "" }
        if candidates.contains("") { return "" }
        return candidates.max(by: { $0.count < $1.count }) ?? ""
    }

    /// A simple trie node used to detect prefix collisions between token sequences.
    private final class PrefixTrieNode {
        var children: [Int: PrefixTrieNode] = [:]
        var isTerminal: Bool = false

        func insertAndCheckCollision(_ sequence: [Int]) -> Bool {
            var current = self

            if sequence.isEmpty {
                if current.isTerminal { return false }
                if !current.children.isEmpty { return true }
                current.isTerminal = true
                return false
            }

            for (index, token) in sequence.enumerated() {
                if current.isTerminal { return true }

                if let child = current.children[token] {
                    current = child
                } else {
                    let child = PrefixTrieNode()
                    current.children[token] = child
                    current = child
                }

                if index == sequence.count - 1 {
                    if current.isTerminal { return false }
                    if !current.children.isEmpty { return true }
                    current.isTerminal = true
                }
            }

            return false
        }
    }

    private static func hasPrefixCollision(tokenized: [[Int]]) -> Bool {
        let root = PrefixTrieNode()
        for sequence in tokenized {
            if root.insertAndCheckCollision(sequence) {
                return true
            }
        }
        return false
    }

    private func compilePattern(_ pattern: String) throws -> NSRegularExpression {
        do {
            return try NSRegularExpression(pattern: pattern)
        } catch {
            throw ConstrainedGenerationError.patternMismatch("Invalid pattern '\(pattern)'")
        }
    }

    private func matchesPattern(_ value: String, regex: NSRegularExpression) -> Bool {
        let range = NSRange(value.startIndex..., in: value)
        return regex.firstMatch(in: value, range: range) != nil
    }
}

// MARK: - Errors

/// An error that can occur during constrained JSON generation.
enum ConstrainedGenerationError: LocalizedError {
    /// A required value failed to tokenize.
    case tokenizationFailed

    /// The generation exceeded the available token budget.
    case tokenBudgetExceeded

    /// The tokenizer does not support a required single-token encoding.
    ///
    /// The associated value contains a user-facing description.
    case unsupportedTokenizer(String)

    /// The generated value does not match the required pattern.
    ///
    /// The associated value contains a user-facing description.
    case patternMismatch(String)

    /// The generated number violates numeric bounds or is invalid.
    ///
    /// The associated value contains a user-facing description.
    case numberOutOfRange(String)

    /// The backend emitted an end token before completion.
    ///
    /// The associated value contains the partial output.
    case earlyTermination(String)

    /// The array bounds are invalid.
    ///
    /// The associated value contains a user-facing description.
    case invalidArrayBounds(String)

    /// A referenced schema definition is missing.
    case missingReference(String)

    /// An any-of schema has no choices.
    case emptyAnyOf

    var errorDescription: String? {
        switch self {
        case .tokenizationFailed:
            return "Failed to tokenize a required value"
        case .tokenBudgetExceeded:
            return "Generation exceeded the available token budget"
        case .unsupportedTokenizer(let details):
            return details
        case .patternMismatch(let details):
            return details
        case .numberOutOfRange(let details):
            return details
        case .earlyTermination:
            return "End token was generated before completion"
        case .invalidArrayBounds(let details):
            return details
        case .missingReference(let name):
            return "Missing referenced schema definition '\(name)'"
        case .emptyAnyOf:
            return "Any-of schema has no choices"
        }
    }
}
