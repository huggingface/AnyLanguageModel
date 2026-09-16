import Testing
@testable import AnyLanguageModel

#if canImport(FoundationModels)
    import struct FoundationModels.GenerationOptions

    @Suite("FoundationModels GenerationOptions")
    struct FoundationModelsGenerationOptionsTests {
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test(
            "Forwards sampling modes",
            arguments: [
                (
                    AnyLanguageModel.GenerationOptions.SamplingMode.greedy,
                    FoundationModels.GenerationOptions.SamplingMode.greedy
                ),
                (
                    AnyLanguageModel.GenerationOptions.SamplingMode.random(top: 40, seed: 123),
                    FoundationModels.GenerationOptions.SamplingMode.random(top: 40, seed: 123)
                ),
                (
                    AnyLanguageModel.GenerationOptions.SamplingMode.random(
                        probabilityThreshold: 0.9,
                        seed: 456
                    ),
                    FoundationModels.GenerationOptions.SamplingMode.random(
                        probabilityThreshold: 0.9,
                        seed: 456
                    )
                ),
            ]
        )
        func forwardsSamplingMode(
            sampling: AnyLanguageModel.GenerationOptions.SamplingMode,
            expected: FoundationModels.GenerationOptions.SamplingMode
        ) {
            let converted = AnyLanguageModel.GenerationOptions(sampling: sampling).toFoundationModels()

            #if compiler(>=6.4) && !os(tvOS)
                #expect(converted.samplingMode == expected)
            #else
                #expect(converted.sampling == expected)
            #endif
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test("Forwards temperature and maximum response tokens")
        func forwardsTemperatureAndMaximumResponseTokens() {
            let converted = AnyLanguageModel.GenerationOptions(
                temperature: 0.5,
                maximumResponseTokens: 42
            ).toFoundationModels()

            #expect(converted.temperature == 0.5)
            #expect(converted.maximumResponseTokens == 42)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test("Preserves unspecified options")
        func preservesUnspecifiedOptions() {
            let converted = AnyLanguageModel.GenerationOptions().toFoundationModels()

            #if compiler(>=6.4) && !os(tvOS)
                #expect(converted.samplingMode == nil)
            #else
                #expect(converted.sampling == nil)
            #endif
            #expect(converted.temperature == nil)
            #expect(converted.maximumResponseTokens == nil)
        }
    }
#endif
