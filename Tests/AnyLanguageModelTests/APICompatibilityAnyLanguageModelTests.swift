import Testing

#if canImport(FoundationModels) && !os(watchOS)
    import AnyLanguageModel

    private let isSystemLanguageModelAvailable: Bool = {
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return SystemLanguageModel.default.isAvailable
        }
        return false
    }()

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private struct CompatibilityDynamicInstructions: DynamicInstructions {
        let includeDetail: Bool

        var body: some DynamicInstructions {
            Instructions("You are a helpful assistant.")
            if includeDetail {
                Instructions("Include useful detail.")
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test("AnyLanguageModel Drop-In Compatibility", .enabled(if: isSystemLanguageModelAvailable))
    func anyLanguageModelCompatibility() async throws {
        let model = SystemLanguageModel.default
        let session = LanguageModelSession(
            model: model,
            instructions: Instructions("You are a helpful assistant.")
        )

        if #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) {
            _ = LanguageModelSession(
                model: model,
                dynamicInstructions: CompatibilityDynamicInstructions(includeDetail: true),
                history: session.transcript
            )
        }

        let options = GenerationOptions(temperature: 0.7)
        let response = try await session.respond(options: options) {
            Prompt("Say 'Hello'")
        }
        #expect(!response.content.isEmpty)

        let stream = session.streamResponse {
            Prompt("Count to 3")
        }
        var hasSnapshots = false
        for try await _ in stream {
            hasSnapshots = true
            break
        }
        #expect(hasSnapshots)
    }
#endif
