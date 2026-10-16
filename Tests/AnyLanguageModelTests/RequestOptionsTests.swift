import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Foundation Models 27 request options")
struct RequestOptionsTests {
    @Test func contextOptionsRetainValues() {
        #expect(ContextOptions() == ContextOptions())
        #expect(
            ContextOptions(includeSchemaInPrompt: true, reasoningLevel: .deep)
                == ContextOptions(includeSchemaInPrompt: true, reasoningLevel: .deep)
        )
        #expect(
            ContextOptions(reasoningLevel: .custom("provider-level"))
                != ContextOptions(reasoningLevel: .moderate)
        )
    }

    @Test(
        arguments: [
            (GenerationOptions.ToolCallingMode.allowed, GenerationOptions.ToolCallingMode.Kind.allowed),
            (.required, .required),
            (.disallowed, .disallowed),
        ]
    )
    func toolCallingModesExposeTheirKind(
        mode: GenerationOptions.ToolCallingMode,
        expected: GenerationOptions.ToolCallingMode.Kind
    ) {
        #expect(mode.kind == expected)
    }

    @Test func generationOptionsKeepOldAndNewSamplingSpellingsInSync() {
        var options = GenerationOptions(samplingMode: .greedy, toolCallingMode: .required)
        #expect(options.sampling == .greedy)
        #expect(options.toolCallingMode == .required)

        options.sampling = .random(top: 4, seed: 7)
        #expect(options.samplingMode == .random(top: 4, seed: 7))

        options.samplingMode = .random(probabilityThreshold: 0.8, seed: 9)
        #expect(options.sampling == .random(probabilityThreshold: 0.8, seed: 9))
    }

    @Test func samplingModeInitializerDoesNotRequireAToolPolicy() {
        let options = GenerationOptions(samplingMode: .greedy)
        #expect(options.sampling == .greedy)
        #expect(options.toolCallingMode == nil)
        #expect(GenerationOptions(samplingMode: nil) == GenerationOptions())
        #expect(
            GenerationOptions(samplingMode: .greedy, temperature: 0.4, maximumResponseTokens: 16)
                == GenerationOptions(sampling: .greedy, temperature: 0.4, maximumResponseTokens: 16)
        )
        #expect(GenerationOptions(temperature: 0.4).temperature == 0.4)
        #expect(GenerationOptions(toolCallingMode: .required).toolCallingMode == .required)
    }

    @Test func transcriptRoundTripsToolCallingMode() throws {
        let options = GenerationOptions(
            samplingMode: .greedy,
            temperature: 0.5,
            maximumResponseTokens: 64,
            toolCallingMode: .required
        )
        let prompt = Transcript.Prompt(
            id: "prompt",
            segments: [.text(.init(id: "text", content: "Hello"))],
            options: options
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(prompt)
        #expect(String(decoding: data, as: UTF8.self).contains(#""toolCallingMode":"required""#))
        let decoded = try JSONDecoder().decode(Transcript.Prompt.self, from: data)
        #expect(decoded.options == options)
    }
}
