/// Preserves omitted wire fields until streaming updates have been merged.
struct ReportedUsage: Sendable {
    struct Input: Sendable {
        var totalTokenCount: Int?

        var cachedTokenCount: Int?

        init(totalTokenCount: Int? = nil, cachedTokenCount: Int? = nil) {
            self.totalTokenCount = totalTokenCount
            self.cachedTokenCount = cachedTokenCount
        }
    }

    struct Output: Sendable {
        var totalTokenCount: Int?

        var reasoningTokenCount: Int?

        init(totalTokenCount: Int? = nil, reasoningTokenCount: Int? = nil) {
            self.totalTokenCount = totalTokenCount
            self.reasoningTokenCount = reasoningTokenCount
        }
    }

    var input: Input

    var output: Output

    init(input: Input = .init(), output: Output = .init()) {
        self.input = input
        self.output = output
    }

    /// Converts omitted wire counts to the public API's zero defaults.
    var value: LanguageModelSession.Usage {
        .init(
            input: .init(totalTokenCount: input.totalTokenCount ?? 0, cachedTokenCount: input.cachedTokenCount ?? 0),
            output: .init(
                totalTokenCount: output.totalTokenCount ?? 0,
                reasoningTokenCount: output.reasoningTokenCount ?? 0
            )
        )
    }

    var isEmpty: Bool {
        input.totalTokenCount == nil && input.cachedTokenCount == nil
            && output.totalTokenCount == nil && output.reasoningTokenCount == nil
    }

    var normalized: Self? { isEmpty ? nil : self }

    /// Adds counts from separate requests,
    /// preserving fields that no request reported.
    mutating func add(_ other: Self?) {
        guard let other else { return }
        func sum(_ lhs: Int?, _ rhs: Int?) -> Int? {
            guard lhs != nil || rhs != nil else { return nil }
            return (lhs ?? 0) + (rhs ?? 0)
        }
        input.totalTokenCount = sum(input.totalTokenCount, other.input.totalTokenCount)
        input.cachedTokenCount = sum(input.cachedTokenCount, other.input.cachedTokenCount)
        output.totalTokenCount = sum(output.totalTokenCount, other.output.totalTokenCount)
        output.reasoningTokenCount = sum(output.reasoningTokenCount, other.output.reasoningTokenCount)
    }

    /// Applies cumulative streaming updates;
    /// omitted fields retain their previous values.
    mutating func merge(_ other: Self?) {
        guard let other else { return }
        input.totalTokenCount = other.input.totalTokenCount ?? input.totalTokenCount
        input.cachedTokenCount = other.input.cachedTokenCount ?? input.cachedTokenCount
        output.totalTokenCount = other.output.totalTokenCount ?? output.totalTokenCount
        output.reasoningTokenCount = other.output.reasoningTokenCount ?? output.reasoningTokenCount
    }
}

/// Usage fields shared by OpenAI's Responses API and Open Responses.
struct ResponsesUsage: Decodable, Sendable {
    let inputTokens: Int?
    let outputTokens: Int?
    let inputTokensDetails: InputDetails?
    let outputTokensDetails: OutputDetails?

    struct InputDetails: Decodable, Sendable {
        let cachedTokens: Int?
        enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" }
    }

    struct OutputDetails: Decodable, Sendable {
        let reasoningTokens: Int?
        enum CodingKeys: String, CodingKey { case reasoningTokens = "reasoning_tokens" }
    }

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case inputTokensDetails = "input_tokens_details"
        case outputTokensDetails = "output_tokens_details"
    }

    var reportedUsage: ReportedUsage? {
        ReportedUsage(
            input: .init(totalTokenCount: inputTokens, cachedTokenCount: inputTokensDetails?.cachedTokens),
            output: .init(totalTokenCount: outputTokens, reasoningTokenCount: outputTokensDetails?.reasoningTokens)
        ).normalized
    }
}

extension LanguageModelSession.ResponseStream.Snapshot {
    /// Builds a snapshot from accumulated text,
    /// including metadata-only updates.
    init?(text: String, usage: LanguageModelSession.Usage) {
        if Content.self == String.self {
            self.init(
                content: (text as! Content).asPartiallyGenerated(),
                rawContent: GeneratedContent(text),
                usage: usage
            )
        } else {
            let raw = (try? GeneratedContent(json: text)) ?? GeneratedContent(text)
            guard let content = try? Content(raw) else { return nil }
            self.init(content: content.asPartiallyGenerated(), rawContent: raw, usage: usage)
        }
    }
}
