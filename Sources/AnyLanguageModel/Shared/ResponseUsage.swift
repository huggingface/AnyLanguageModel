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

    var metadata: [String: GeneratedContent]

    init(input: Input = .init(), output: Output = .init(), metadata: [String: GeneratedContent] = [:]) {
        self.input = input
        self.output = output
        self.metadata = metadata
    }

    /// Converts omitted wire counts to the public API's zero defaults.
    var value: LanguageModelSession.Usage {
        .init(
            input: .init(totalTokenCount: input.totalTokenCount ?? 0, cachedTokenCount: input.cachedTokenCount ?? 0),
            output: .init(
                totalTokenCount: output.totalTokenCount ?? 0,
                reasoningTokenCount: output.reasoningTokenCount ?? 0
            ),
            metadata: metadata
        )
    }

    var isEmpty: Bool {
        input.totalTokenCount == nil && input.cachedTokenCount == nil
            && output.totalTokenCount == nil && output.reasoningTokenCount == nil
            && metadata.isEmpty
    }

    var normalized: Self? { isEmpty ? nil : self }

    /// Adds counts from separate requests,
    /// preserving fields that no request reported.
    /// As with session usage, the latest value wins for each metadata key.
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
        metadata.merge(other.metadata) { _, latest in latest }
    }

    /// Applies cumulative streaming updates;
    /// omitted fields retain their previous values.
    mutating func merge(_ other: Self?) {
        guard let other else { return }
        input.totalTokenCount = other.input.totalTokenCount ?? input.totalTokenCount
        input.cachedTokenCount = other.input.cachedTokenCount ?? input.cachedTokenCount
        output.totalTokenCount = other.output.totalTokenCount ?? output.totalTokenCount
        output.reasoningTokenCount = other.output.reasoningTokenCount ?? output.reasoningTokenCount
        metadata.merge(other.metadata) { _, latest in latest }
    }
}

/// The usage of a response's requests, one for each tool round, each with the conversation so far.
///
/// As in Foundation Models, a response reports its final request's usage,
/// and the session counts every request.
struct RequestsUsage: Sendable, Equatable {
    /// Every request's usage before the final one, added up.
    private(set) var earlier: LanguageModelSession.Usage

    /// The final request's usage.
    var final: LanguageModelSession.Usage

    /// Every request's usage, added up.
    var total: LanguageModelSession.Usage {
        var total = earlier
        total.add(final)
        return total
    }

    init(earlier: LanguageModelSession.Usage = .zero, final: LanguageModelSession.Usage = .zero) {
        self.earlier = earlier
        self.final = final
    }

    /// Requests whose usage is known as a total, with the final request's usage.
    init(final: LanguageModelSession.Usage, total: LanguageModelSession.Usage) {
        self.init(earlier: total.subtracting(final), final: final)
    }

    /// Adds the usage of a request sent after the others, and counts it for the response being
    /// made as it completes.
    mutating func add(_ request: LanguageModelSession.Usage) {
        earlier.add(final)
        final = request
        RequestUsageChannel.report(request)
    }

    /// Adds the usage a request sent after the others reported, if any.
    mutating func add(_ request: ReportedUsage?) {
        add(request?.value ?? .zero)
    }
}

/// Where the requests of a response that isn't streamed are counted as each one completes, as
/// Foundation Models' executor sends each request's usage to its session while the session runs
/// the tool loop. A response that throws keeps the usage of the requests it made.
///
/// The session sets one around each such response. What a tool or a stream asks of a model runs
/// with none, so it's never counted for the response that called the tool or made the stream.
///
/// It's a task-local, so a model reaches it wherever it runs in the response's task, as when a
/// model of another library wraps a built-in one and calls it. A `Task {}` made within the response
/// inherits it, so what it asks of a model counts for the response; `Task.detached` doesn't.
final class RequestUsageChannel: Sendable {
    @TaskLocal static var current: RequestUsageChannel?

    private let record: @Sendable (LanguageModelSession.Usage) -> Void
    private let reported = Locked(LanguageModelSession.Usage.zero)

    /// A channel that passes each request's usage to `record` as it's reported.
    init(record: @escaping @Sendable (LanguageModelSession.Usage) -> Void) {
        self.record = record
    }

    /// Counts the usage of a request that completed, for the response being made, if any.
    static func report(_ usage: LanguageModelSession.Usage) {
        current?.send(usage)
    }

    /// Counts the usage of a request that completed and reported any, for the response being made.
    static func report(_ usage: ReportedUsage?) {
        if let usage { report(usage.value) }
    }

    /// Runs `operation` where no response's requests are counted.
    ///
    /// It runs on the caller's executor. Swift 6.4 adds the `nonisolated(nonsending)` overload
    /// of `withValue` that this calls; earlier versions call the one that takes an isolation.
    nonisolated(nonsending) static func outside<T>(
        _ operation: nonisolated(nonsending) () async throws -> T
    ) async rethrows -> T {
        try await $current.withValue(nil, operation: operation)
    }

    /// Runs `operation` where no response's requests are counted.
    static func outside<T>(_ operation: () throws -> T) rethrows -> T {
        try $current.withValue(nil, operation: operation)
    }

    /// Every request's usage reported so far, added up.
    var reportedUsage: LanguageModelSession.Usage {
        reported.withLock { $0 }
    }

    private func send(_ usage: LanguageModelSession.Usage) {
        guard usage != .zero else { return }
        reported.withLock { $0.add(usage) }
        record(usage)
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
    ///
    /// - Parameter wrapperProperty: The property that holds the response,
    ///   if the provider's response schema wraps it.
    init?(text: String, requests: RequestsUsage, wrapperProperty: String? = nil) {
        if Content.self == String.self {
            self.init(
                content: (text as! Content).asPartiallyGenerated(),
                rawContent: GeneratedContent(text),
                requests: requests
            )
        } else {
            let raw = ProviderResponseSchema.unwrapping(
                (try? GeneratedContent(json: text)) ?? GeneratedContent(text),
                fromProperty: wrapperProperty
            )
            guard let content = partiallyGenerated(Content.self, from: raw) else { return nil }
            self.init(content: content, rawContent: raw, requests: requests)
        }
    }
}

/// Decodes partially generated content, or returns `nil`.
///
/// A generic function, because Swift 6.3 can't type-check
/// this initializer inside the snapshot's extension.
private func partiallyGenerated<Value: Generable>(
    _: Value.Type,
    from raw: GeneratedContent
) -> Value.PartiallyGenerated? {
    try? Value.PartiallyGenerated(raw)
}
