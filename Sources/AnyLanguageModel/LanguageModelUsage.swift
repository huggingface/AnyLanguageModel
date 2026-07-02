/// Provider-reported token usage for a language model response.
///
/// Values are optional because not every provider or endpoint reports every field.
/// When a single `respond` call internally triggers multiple provider requests
/// (for example while resolving tool calls), the returned usage is aggregated
/// across those underlying requests.
public struct LanguageModelUsage: Hashable, Codable, Sendable {
    /// Tokens consumed by the request input or prompt.
    public var inputTokens: Int?

    /// Tokens generated in the response.
    public var outputTokens: Int?

    /// Total tokens reported for the request.
    public var totalTokens: Int?

    /// Tokens spent on reasoning or thinking, when reported separately.
    public var reasoningTokens: Int?

    /// Input tokens served from prompt cache, when reported separately.
    public var cachedInputTokens: Int?

    /// Input tokens written into a prompt cache, when reported separately.
    public var cacheCreationInputTokens: Int?

    public init(
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        totalTokens: Int? = nil,
        reasoningTokens: Int? = nil,
        cachedInputTokens: Int? = nil,
        cacheCreationInputTokens: Int? = nil
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.totalTokens = totalTokens
        self.reasoningTokens = reasoningTokens
        self.cachedInputTokens = cachedInputTokens
        self.cacheCreationInputTokens = cacheCreationInputTokens
    }
}

extension LanguageModelUsage {
    var isEmpty: Bool {
        inputTokens == nil
            && outputTokens == nil
            && totalTokens == nil
            && reasoningTokens == nil
            && cachedInputTokens == nil
            && cacheCreationInputTokens == nil
    }

    var normalized: Self? {
        isEmpty ? nil : self
    }

    mutating func add(_ other: Self?) {
        guard let other else { return }
        inputTokens = Self.sum(inputTokens, other.inputTokens)
        outputTokens = Self.sum(outputTokens, other.outputTokens)
        totalTokens = Self.sum(totalTokens, other.totalTokens)
        reasoningTokens = Self.sum(reasoningTokens, other.reasoningTokens)
        cachedInputTokens = Self.sum(cachedInputTokens, other.cachedInputTokens)
        cacheCreationInputTokens = Self.sum(cacheCreationInputTokens, other.cacheCreationInputTokens)
    }

    mutating func merge(_ other: Self?) {
        guard let other else { return }
        inputTokens = other.inputTokens ?? inputTokens
        outputTokens = other.outputTokens ?? outputTokens
        totalTokens = other.totalTokens ?? totalTokens
        reasoningTokens = other.reasoningTokens ?? reasoningTokens
        cachedInputTokens = other.cachedInputTokens ?? cachedInputTokens
        cacheCreationInputTokens = other.cacheCreationInputTokens ?? cacheCreationInputTokens
    }

    private static func sum(_ lhs: Int?, _ rhs: Int?) -> Int? {
        switch (lhs, rhs) {
        case (.some(let lhs), .some(let rhs)):
            lhs + rhs
        case (.some(let lhs), .none):
            lhs
        case (.none, .some(let rhs)):
            rhs
        case (.none, .none):
            nil
        }
    }
}
