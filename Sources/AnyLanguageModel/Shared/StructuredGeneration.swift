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

/// Last-resort token-budget floor for optional structure (object keys / array items).
///
/// Optional *selection* is model-driven (see ``ConstrainedJSONGenerator``).
/// This floor only kicks in when generation is about to run out of tokens: once the
/// schema-valid minimum has been emitted (all required keys, or `minItems` elements),
/// the generator stops offering further optionals and closes rather than risking a
/// hard budget failure mid-value.
private enum OptionalStructureBudget {
    /// Minimum absolute number of tokens that should remain before offering more
    /// optional properties or array elements.
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

    private let tokensByKey = Locked<[Key: Set<Int>]>([:])

    func tokens(for key: Key) -> Set<Int>? {
        tokensByKey.withLock { $0[key] }
    }

    func store(_ tokens: Set<Int>, for key: Key) {
        tokensByKey.withLock { $0[key] = tokens }
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

    /// ASCII digit `0`...`9` only — JSON numbers must not accept fullwidth / superscript
    /// forms that `Character.isNumber` would otherwise admit.
    private static func isASCIIDigit(_ character: Character) -> Bool {
        character >= "0" && character <= "9"
    }

    /// Tokens that may appear inside a JSON integer: ASCII digits and standalone `-`.
    ///
    /// Standalone `-` is required because BPE tokenizers (Qwen2.5, etc.) encode `-1` as
    /// two tokens. Requiring every token to contain a digit excluded `-` and made negatives
    /// unrepresentable except via rare multi-character tokens.
    private static func buildValidIntegerTokens(backend: Backend) -> Set<Int> {
        var allowed = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            let onlyIntegerChars = text.allSatisfy { Self.isASCIIDigit($0) || $0 == "-" }
            let hasDigit = text.contains { Self.isASCIIDigit($0) }
            let isStandaloneMinus = text == "-"
            if onlyIntegerChars && (hasDigit || isStandaloneMinus) {
                allowed.insert(token)
            }
        }
        return allowed
    }

    /// Tokens that may appear inside a JSON number: ASCII digits, `-`, and `.`.
    ///
    /// **Critical:** standalone `.` and `-` must be included. Qwen2.5 encodes `473.00` as
    /// `4` `7` `3` `.` `0` `0`. The previous filter required every token to contain a digit,
    /// which dropped `.` and forced the model to pad zeros until `maxDecimalTokenLimit`
    /// (pathological `e+31` values after Double re-serialization).
    private static func buildValidDecimalTokens(backend: Backend) -> Set<Int> {
        var allowed = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token), !text.isEmpty else { continue }
            let onlyNumberChars = text.allSatisfy {
                Self.isASCIIDigit($0) || $0 == "-" || $0 == "."
            }
            let hasDigit = text.contains { Self.isASCIIDigit($0) }
            let isStandaloneSignOrDot = text == "-" || text == "."
            if onlyNumberChars && (hasDigit || isStandaloneSignOrDot) {
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
        // Object *key set* is model-driven under the JSON grammar. The previous
        // implementation pre-filtered optional properties with a hash of the field
        // name XOR the token budget, so each optional was always-on or always-off
        // for a given budget — decided before the model was consulted.
        //
        // After each property the mask permits any not-yet-emitted key, and permits
        // `}` once every required property has been emitted. For schemas with no
        // required properties that makes `{}` reachable (schema-valid, and what the
        // model asked for) — a visible behaviour change for such schemas.
        var remainingKeys = Set(node.properties.keys)
        let required = node.required
        var output = try await emit("{")
        var emittedAnyProperty = false

        while !remainingKeys.isEmpty {
            let missingRequired = required.intersection(remainingKeys)
            let canClose = missingRequired.isEmpty
            let budgetAllowsMoreOptionals = hasBudgetForOptionalStructure()

            let keysToOffer: [String]
            if budgetAllowsMoreOptionals {
                // Model chooses any not-yet-emitted property (required or optional).
                keysToOffer = remainingKeys.sorted()
            } else if canClose {
                // Genuine last resort: stop offering optionals when the budget is nearly
                // exhausted. Required properties are already present, so close cleanly.
                break
            } else {
                // Still missing required keys — only those may be emitted under pressure.
                keysToOffer = missingRequired.sorted()
            }

            var candidates: [String] = keysToOffer.map { key in
                let prefix = emittedAnyProperty ? "," : ""
                return "\(prefix)\"\(key)\":"
            }
            // Closing is legal once every required property has been emitted. The model
            // may leave remaining optionals out; that is schema-valid JSON.
            if canClose {
                candidates.append("}")
            }

            guard !candidates.isEmpty else { break }

            let choice = try await generateChoice(candidates)
            output += choice

            // Model elected to close; remaining keys are optional and intentionally omitted.
            if choice == "}" {
                return output
            }

            guard let key = propertyKey(fromPropertyStart: choice),
                let valueNode = node.properties[key]
            else {
                throw ConstrainedGenerationError.tokenizationFailed
            }
            remainingKeys.remove(key)
            output += try await generateNode(valueNode)
            emittedAnyProperty = true
        }

        output += try await emit("}")
        return output
    }

    private mutating func generateArray(_ node: GenerationSchema.ArrayNode) async throws -> String {
        // Array *length* is model-driven under the JSON grammar — same family of fix as
        // model-driven optional object keys. After each element the mask permits both
        // continuing (`,`) and closing (`]`), subject to schema `minItems` / `maxItems`.
        // A budget-derived fixed count (or `totalTokenBudget % rangeSize`) would force the
        // same length for every document and invent filler or truncate real items.
        let minItems = max(0, node.minItems ?? 0)
        let maxItems = node.maxItems

        if let maxItems, minItems > maxItems {
            throw ConstrainedGenerationError.invalidArrayBounds(
                "Minimum items \(minItems) exceeds maximum \(maxItems)"
            )
        }

        var output = try await emit("[")
        var count = 0

        while true {
            if let maxItems, count >= maxItems {
                break
            }

            let canClose = count >= minItems
            let budgetAllowsMore = hasBudgetForOptionalStructure()

            if canClose && !budgetAllowsMore {
                // Genuine last resort: close once minItems is satisfied rather than
                // failing mid-element under a hard budget floor.
                break
            }

            if count > 0 {
                if canClose {
                    // Model chooses continue vs close.
                    let choice = try await generateChoice([",", "]"])
                    if choice == "]" {
                        output += choice
                        return output
                    }
                    output += choice
                } else {
                    // Still below minItems — must emit another element.
                    output += try await emit(",")
                }
            } else if canClose {
                // Empty array is legal (`minItems == 0`). Probe whether the model wants
                // `]` or a first element. Sampling is non-committing for non-`]` tokens
                // (see ``sampleWhetherToCloseEmptyArray``); the element is then generated
                // from the same decode state.
                if try await sampleWhetherToCloseEmptyArray(items: node.items) {
                    output += try await emit("]")
                    return output
                }
            }

            output += try await generateNode(node.items)
            count += 1
        }

        output += try await emit("]")
        return output
    }

    /// Probe after `[` when `minItems == 0`: model may close immediately or start an item.
    ///
    /// Samples once among closing tokens and tokens that can start the item type. Choosing
    /// a close means close; any other sample is discarded without decoding so
    /// ``generateNode`` can emit the first element from the same backend state.
    ///
    /// Byte-pair vocabularies put most item-start probability on merged tokens (a quote
    /// fused with the first word, or a leading space), so the probe admits every token
    /// whose first non-whitespace character starts the item type, not just the bare
    /// single-character token.
    private mutating func sampleWhetherToCloseEmptyArray(
        items: GenerationSchema.Node
    ) async throws -> Bool {
        let closeTokens = tokensMatchingTrimmed("]")
        var allowed = try itemStartTokens(for: items)
        allowed.formUnion(closeTokens)
        guard !allowed.isEmpty, !closeTokens.isEmpty else {
            return false
        }
        let token = try await backend.sample(from: allowed)
        return closeTokens.contains(token)
    }

    /// Tokens whose text equals `text` after trimming surrounding whitespace.
    private func tokensMatchingTrimmed(_ text: String) -> Set<Int> {
        var tokens = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let tokenText = backend.tokenText(token) else { continue }
            if tokenText.trimmingCharacters(in: .whitespacesAndNewlines) == text {
                tokens.insert(token)
            }
        }
        return tokens
    }

    /// Tokens whose text, after leading whitespace, is a non-empty prefix of one of `literals`.
    private func tokensPrefixing(anyOf literals: [String]) -> Set<Int> {
        var tokens = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token) else { continue }
            let trimmed = text.drop(while: { $0.isWhitespace })
            guard !trimmed.isEmpty else { continue }
            if literals.contains(where: { $0.hasPrefix(trimmed) }) {
                tokens.insert(token)
            }
        }
        return tokens
    }

    /// Tokens whose first non-whitespace character is `prefix`.
    private func tokensStarting(with prefix: Character) -> Set<Int> {
        var tokens = Set<Int>()
        for token in 0 ..< backend.vocabSize {
            if backend.isSpecialToken(token) { continue }
            guard let text = backend.tokenText(token) else { continue }
            guard let first = text.drop(while: { $0.isWhitespace }).first else { continue }
            if first == prefix {
                tokens.insert(token)
            }
        }
        return tokens
    }

    /// Tokens that can begin a JSON value for `node` (empty-array probe).
    private func itemStartTokens(for node: GenerationSchema.Node) throws -> Set<Int> {
        switch node {
        case .string:
            return tokensStarting(with: "\"")
        case .object:
            return tokensStarting(with: "{")
        case .array:
            return tokensStarting(with: "[")
        case .boolean:
            return tokensPrefixing(anyOf: ["true", "false"])
        case .number(let numberNode):
            let numeric =
                numberNode.integerOnly
                ? integerTerminators.subtracting(basicTerminators)
                : doubleTerminators.subtracting(basicTerminators)
            // Only tokens that can start a number (digit or minus — not a bare `.`).
            return Set(
                numeric.filter { token in
                    guard let text = backend.tokenText(token),
                        let first = text.drop(while: { $0.isWhitespace }).first
                    else { return false }
                    return first.isNumber || first == "-"
                }
            )
        case .ref(let typeName):
            guard let referenced = schema.defs[typeName] else {
                throw ConstrainedGenerationError.missingReference(typeName)
            }
            return try itemStartTokens(for: referenced)
        case .anyOf(let variants):
            var tokens = Set<Int>()
            for variant in variants {
                tokens.formUnion(try itemStartTokens(for: variant))
            }
            return tokens
        }
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

    /// Whether enough budget remains to *offer* more optional structure to the model
    /// (object properties or array elements).
    ///
    /// This is a last-resort guard only. It does not pick which optionals appear or how
    /// long an array is — those decisions are made by constrained sampling.
    private func hasBudgetForOptionalStructure() -> Bool {
        let minimumBudget = OptionalStructureBudget.minimumBudget(
            totalTokenBudget: backend.totalTokenBudget
        )
        return backend.remainingTokens > minimumBudget
    }

    /// Parses a property-start fragment produced for object key selection.
    ///
    /// Expected shapes: `"key":` (first property) or `,"key":` (subsequent).
    private func propertyKey(fromPropertyStart choice: String) -> String? {
        var fragment = choice
        if fragment.first == "," {
            fragment.removeFirst()
        }
        guard fragment.first == "\"", fragment.hasSuffix("\":") else { return nil }
        fragment.removeFirst()
        fragment.removeLast(2)
        return fragment
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
