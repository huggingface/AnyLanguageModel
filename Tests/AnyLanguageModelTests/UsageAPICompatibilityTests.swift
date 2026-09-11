import Testing

// Define FOUNDATION_MODELS_USAGE_COMPATIBILITY with an OS 27 SDK
// to compile these same API checks against Apple's framework.
#if FOUNDATION_MODELS_USAGE_COMPATIBILITY
    import FoundationModels
#else
    import AnyLanguageModel
#endif

#if FOUNDATION_MODELS_USAGE_COMPATIBILITY
    @available(macOS 27.0, iOS 27.0, watchOS 27.0, tvOS 27.0, visionOS 27.0, *)
#endif
@Suite("Foundation Models 27 usage API compatibility")
struct UsageAPICompatibilityTests {
    @Test func publicInitializersAndProperties() {
        typealias Usage = LanguageModelSession.Usage
        let makeInput: (Int, Int) -> Usage.Input = Usage.Input.init(totalTokenCount:cachedTokenCount:)
        let makeOutput: (Int, Int) -> Usage.Output = Usage.Output.init(totalTokenCount:reasoningTokenCount:)
        let makeUsage: (Usage.Input, Usage.Output, [String: any ConvertibleToGeneratedContent]) -> Usage =
            Usage.init(input:output:metadata:)

        var usage = makeUsage(makeInput(100, 25), makeOutput(20, 5), ["service_tier": "standard"])
        let initialTotal: Int = usage.totalTokenCount
        #expect(initialTotal == 120)
        let metadata: [String: GeneratedContent] = usage.metadata
        #expect(metadata["service_tier"]?.jsonString == "\"standard\"")

        usage.input.totalTokenCount = 200
        usage.input.cachedTokenCount = 50
        usage.output.totalTokenCount = 40
        usage.output.reasoningTokenCount = 10
        usage.metadata = [:]
        #expect(usage.totalTokenCount == 240)
        #expect(usage.metadata.isEmpty)

        let withoutMetadata = Usage(input: makeInput(0, 0), output: makeOutput(0, 0))
        #expect(withoutMetadata.metadata.isEmpty)
        #expect(withoutMetadata.totalTokenCount == 0)
    }

    @Test func usageIsNonOptionalOnSessionsResponsesAndSnapshots() {
        typealias Usage = LanguageModelSession.Usage
        let _: KeyPath<LanguageModelSession, Usage> = \.usage
        let _: KeyPath<LanguageModelSession.Response<String>, Usage> = \.usage
        let _: WritableKeyPath<LanguageModelSession.ResponseStream<String>.Snapshot, Usage> = \.usage
    }
}
