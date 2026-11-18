import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    import LlamaSwift

    /// Tests of the sampler chain that `LlamaLanguageModel` builds,
    /// applied to hand-built candidates without a model.
    @Suite("Llama sampler chain")
    struct LlamaSamplerChainTests {
        /// Mirostat v1 keeps the most probable tokens, and the number that it keeps
        /// depends on the vocabulary size that the sampler gets.
        @Test("Mirostat v1 gets the vocabulary size")
        func mirostatV1UsesVocabularySize() throws {
            let vocabularySize: Int32 = 32_000
            let contextSize: UInt32 = 2048
            let options = LlamaLanguageModel.ResolvedGenerationOptions(
                contextSize: contextSize,
                seed: 1,
                temperature: 1,
                repeatPenalty: 1,
                mirostat: .v1(tau: 5, eta: 0.1)
            )

            let chain = try #require(
                LlamaLanguageModel.makeSamplerChain(vocabularySize: vocabularySize, options: options)
            )
            defer { llama_sampler_free(chain) }
            let expected = try #require(llama_sampler_init_mirostat(vocabularySize, 1, 5, 0.1, 100))
            defer { llama_sampler_free(expected) }
            let withContextSize = try #require(llama_sampler_init_mirostat(Int32(contextSize), 1, 5, 0.1, 100))
            defer { llama_sampler_free(withContextSize) }

            let keptCount = candidateCount(after: chain, vocabularySize: vocabularySize)
            #expect(keptCount == candidateCount(after: expected, vocabularySize: vocabularySize))
            #expect(keptCount != candidateCount(after: withContextSize, vocabularySize: vocabularySize))
        }

        /// Applies a sampler to Zipf-distributed candidates
        /// and returns the number of candidates that it keeps.
        private func candidateCount(after sampler: UnsafeMutablePointer<llama_sampler>, vocabularySize: Int32) -> Int {
            var candidates = (0 ..< vocabularySize).map { id in
                llama_token_data(id: id, logit: -1.1 * logf(Float(id + 1)), p: 0)
            }
            return candidates.withUnsafeMutableBufferPointer { buffer in
                var array = llama_token_data_array(
                    data: buffer.baseAddress,
                    size: buffer.count,
                    selected: -1,
                    sorted: false
                )
                llama_sampler_apply(sampler, &array)
                return array.size
            }
        }
    }
#endif
