import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    import LlamaSwift

    @Suite(
        "LlamaLanguageModel",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil)
    )
    struct LlamaLanguageModelTests {
        let model = LlamaLanguageModel(
            modelPath: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"]!
        )

        @Test func initialization() {
            let customModel = LlamaLanguageModel(modelPath: "/path/to/model.gguf")
            #expect(customModel.localModelPath == "/path/to/model.gguf")
            #expect(customModel.gpuLayers == LlamaLanguageModel.defaultGPULayerCount)

            let cpuOnlyModel = LlamaLanguageModel(modelPath: "/path/to/model.gguf", gpuLayers: 0)
            #expect(cpuOnlyModel.gpuLayers == 0)
        }

        /// A cancelled request throws, and no response goes into the transcript.
        @Test func cancellationBeforeGenerationThrows() async throws {
            let session = LanguageModelSession(model: model)
            let task = Task {
                try await session.respond(to: "Say hello", options: GenerationOptions(maximumResponseTokens: 16))
            }
            task.cancel()
            await #expect(throws: CancellationError.self) { try await task.value }
            #expect(
                !session.transcript.contains { entry in
                    if case .response = entry { true } else { false }
                }
            )
        }

        @Test func cancellationDuringGenerationThrows() async throws {
            let session = LanguageModelSession(model: model)
            // Load the model first, so that the cancellation comes during generation.
            _ = try await LanguageModelSession(model: model).respond(
                to: "Say hello",
                options: GenerationOptions(maximumResponseTokens: 16)
            )
            let task = Task {
                try await session.respond(
                    to: "Count from 1 to 1000, one number per line.",
                    options: GenerationOptions(maximumResponseTokens: 4000)
                )
            }
            try await Task.sleep(for: .milliseconds(500))
            task.cancel()
            await #expect(throws: CancellationError.self) { try await task.value }
            #expect(
                !session.transcript.contains { entry in
                    if case .response = entry { true } else { false }
                }
            )
        }

        @Test func concurrentFirstRequests() async throws {
            try await withThrowingTaskGroup(of: String.self) { group in
                for _ in 0 ..< 2 {
                    group.addTask {
                        let session = LanguageModelSession(model: model)
                        let response = try await session.respond(
                            to: "Reply with a single word.",
                            options: GenerationOptions(maximumResponseTokens: 16)
                        )
                        attachResponse(response)
                        return response.content
                    }
                }

                var responseCount = 0
                for try await content in group {
                    #expect(!content.isEmpty)
                    responseCount += 1
                }
                #expect(responseCount == 2)
            }
        }

        @Test func promptLongerThanBatchSize() async throws {
            let session = LanguageModelSession(model: model)
            var options = GenerationOptions(maximumResponseTokens: 16)
            options[custom: LlamaLanguageModel.self] = .init(batchSize: 32)

            let filler = Array(
                repeating: "The quick brown fox jumps over the lazy dog.",
                count: 30
            ).joined(separator: " ")
            let response = try await session.respond(
                to: "\(filler)\n\nReply with a single word.",
                options: options
            )
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        @Test func reusesSessionContextAcrossTurns() async throws {
            let session = LanguageModelSession(model: model)
            var options = GenerationOptions(maximumResponseTokens: 24)
            options[custom: LlamaLanguageModel.self] = .init(contextSize: 2048, batchSize: 512)

            let first = try await session.respond(
                to: "My favorite color is blue. Reply with OK.",
                options: options
            )
            attachResponse(first, "first")
            #expect(!first.content.isEmpty)
            #expect(model.lastReusedTokenCount == 0)
            #expect(first.usage.input.cachedTokenCount == 0)
            #expect(first.usage.input.totalTokenCount == model.lastPrefillTokenCount)
            #expect(first.usage.output.totalTokenCount > 0)

            let second = try await session.respond(
                to: "What is my favorite color? Answer with one word.",
                options: options
            )
            attachResponse(second, "second")
            #expect(!second.content.isEmpty)
            #expect(model.lastReusedTokenCount > 0)
            #expect(second.usage.input.cachedTokenCount == model.lastReusedTokenCount)
            #expect(second.usage.input.totalTokenCount == model.lastReusedTokenCount + model.lastPrefillTokenCount)
            #expect(session.usage.totalTokenCount == first.usage.totalTokenCount + second.usage.totalTokenCount)

            model.clearCachedContext()
            let afterClear = try await session.respond(to: "Say OK.", options: options)
            attachResponse(afterClear, "afterClear")
            #expect(afterClear.usage.input.cachedTokenCount == 0)
            #expect(afterClear.usage.input.totalTokenCount == model.lastPrefillTokenCount)
        }

        @Test func customGenerationOptionsRoundTrip() {
            var options = GenerationOptions(
                temperature: 0.6,
                maximumResponseTokens: 25
            )

            let custom = LlamaLanguageModel.CustomGenerationOptions(
                contextSize: 1024,
                batchSize: 256,
                threads: 1,
                seed: 42,
                temperature: 0.55,
                topK: 25,
                topP: 0.85,
                repeatPenalty: 1.15,
                repeatLastN: 48,
                frequencyPenalty: 0.05,
                presencePenalty: 0.05,
                mirostat: .v2(tau: 5.0, eta: 0.2),
                assistantPrefill: "<think></think>"
            )
            options[custom: LlamaLanguageModel.self] = custom

            let retrieved = options[custom: LlamaLanguageModel.self]
            #expect(retrieved?.contextSize == 1024)
            #expect(retrieved?.batchSize == 256)
            #expect(retrieved?.threads == 1)
            #expect(retrieved?.seed == 42)
            #expect(retrieved?.temperature == 0.55)
            #expect(retrieved?.topK == 25)
            #expect(retrieved?.topP == 0.85)
            #expect(retrieved?.repeatPenalty == 1.15)
            #expect(retrieved?.repeatLastN == 48)
            #expect(retrieved?.frequencyPenalty == 0.05)
            #expect(retrieved?.presencePenalty == 0.05)
            #expect(retrieved?.mirostat == .v2(tau: 5.0, eta: 0.2))
            #expect(retrieved?.assistantPrefill == "<think></think>")
        }

        @Test func customGenerationOptionsDefaults() {
            let defaults = LlamaLanguageModel.CustomGenerationOptions.default
            #expect(defaults.contextSize == 2048)
            #expect(defaults.batchSize == 512)
            #expect(defaults.threads == Int32(ProcessInfo.processInfo.processorCount))
            #expect(defaults.seed == nil)
            #expect(defaults.temperature == 0.8)
            #expect(defaults.topK == 40)
            #expect(defaults.topP == 0.95)
            #expect(defaults.repeatPenalty == 1.1)
            #expect(defaults.repeatLastN == 64)
            #expect(defaults.frequencyPenalty == 0.0)
            #expect(defaults.presencePenalty == 0.0)
            #expect(defaults.mirostat == nil)
        }

        @Test func logLevelConfiguration() {
            let originalLevel = LlamaLanguageModel.logLevel

            LlamaLanguageModel.logLevel = .none
            #expect(LlamaLanguageModel.logLevel == .none)

            LlamaLanguageModel.logLevel = .debug
            #expect(LlamaLanguageModel.logLevel == .debug)

            LlamaLanguageModel.logLevel = .error
            #expect(LlamaLanguageModel.logLevel == .error)

            LlamaLanguageModel.logLevel = originalLevel
        }

        @Test func logLevelComparison() {
            #expect(LlamaLanguageModel.LogLevel.none < .debug)
            #expect(LlamaLanguageModel.LogLevel.debug < .info)
            #expect(LlamaLanguageModel.LogLevel.info < .warn)
            #expect(LlamaLanguageModel.LogLevel.warn < .error)

            #expect(LlamaLanguageModel.LogLevel.error > .warn)
            #expect(LlamaLanguageModel.LogLevel.warn >= .warn)
        }

        @Test func logLevelHashable() {
            let levels: Set<LlamaLanguageModel.LogLevel> = [.debug, .info, .warn]
            #expect(levels.contains(.debug))
            #expect(levels.contains(.info))
            #expect(levels.contains(.warn))
            #expect(!levels.contains(.none))
            #expect(!levels.contains(.error))
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )

            let response = try await session.respond(to: "What is 2+2?")
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        @Test func tokenUsageResponseStreamParity() async throws {
            var options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 8)
            options[custom: LlamaLanguageModel.self] = .init(seed: 42)
            let response = try await LanguageModelSession(model: model).respond(to: "Say hello.", options: options)
            attachResponse(response)
            let session = LanguageModelSession(model: model)
            let streamed = try await session.streamResponse(to: "Say hello.", options: options).collect()
            #expect(response.usage.input.totalTokenCount > 0)
            #expect(response.usage.output.totalTokenCount > 0)
            #expect(response.usage.input.cachedTokenCount == 0)
            #expect(streamed.usage == response.usage)
            #expect(streamed.content == response.content)
            #expect(session.usage == streamed.usage)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            attachResponse(firstResponse, "firstResponse")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            attachResponse(secondResponse, "secondResponse")
            #expect(!secondResponse.content.isEmpty)
        }

        @Test func maxTokensLimit() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(maximumResponseTokens: 10)
            let response = try await session.respond(
                to: "Write a long essay about artificial intelligence",
                options: options
            )
            attachResponse(response)

            // Response should be limited by max tokens
            #expect(!response.content.isEmpty)
        }

        @Test func greedySamplingWithTemperature() async throws {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(
                sampling: .greedy,
                temperature: 0.7,
                maximumResponseTokens: 50
            )
            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.8,
                maximumResponseTokens: 50
            )

            // Set llama.cpp-specific custom options
            options[custom: LlamaLanguageModel.self] = .init(
                contextSize: 1024,
                batchSize: 256,
                threads: 2,
                seed: 123,
                temperature: 0.75,
                topK: 30,
                topP: 0.9,
                repeatPenalty: 1.2,
                repeatLastN: 128,
                frequencyPenalty: 0.1,
                presencePenalty: 0.1
            )

            let response = try await session.respond(
                to: "Tell me a short fact",
                options: options
            )
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        @Test func withMirostatSampling() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.8,
                maximumResponseTokens: 50
            )

            // Use mirostat v2 for adaptive perplexity control
            options[custom: LlamaLanguageModel.self] = .init(
                mirostat: .v2(tau: 5.0, eta: 0.1)
            )

            let response = try await session.respond(
                to: "Tell me a short fact",
                options: options
            )
            attachResponse(response)
            #expect(!response.content.isEmpty)
        }

        /// The token loop accepts each sampled token into the sampler chain once,
        /// so it makes the same tokens as a loop that calls `llama_sampler_sample` alone.
        /// The penalties sampler counts accepted tokens, so a second accept changes them.
        @Test func tokenLoopAcceptsEachTokenOnce() async throws {
            try await model.ensureModelLoaded()
            let llamaModel = try #require(model.model)
            let vocab = try #require(llama_model_get_vocab(llamaModel))
            let options = LlamaLanguageModel.ResolvedGenerationOptions(
                seed: 42,
                repeatPenalty: 1.3,
                repeatLastN: 64,
                frequencyPenalty: 0.5,
                presencePenalty: 0.5
            )
            let maxTokens = 64
            let prompt = "List the colors of the rainbow, then list them again:"
            let promptTokens = try #require(tokenize(prompt, vocab: vocab))

            let packageContext = try #require(makeContext(llamaModel, options: options))
            defer { llama_free(packageContext) }
            var packagePieces: [String] = []
            _ = try model.performTokenGeneration(
                context: packageContext,
                vocab: vocab,
                promptTokens: promptTokens,
                startIndex: 0,
                maxTokens: maxTokens,
                options: options
            ) { piece in
                packagePieces.append(piece)
                return true
            }

            let referenceContext = try #require(makeContext(llamaModel, options: options))
            defer { llama_free(referenceContext) }
            let sampler = try #require(
                LlamaLanguageModel.makeSamplerChain(vocabularySize: llama_vocab_n_tokens(vocab), options: options)
            )
            defer { llama_sampler_free(sampler) }
            var batch = llama_batch_init(Int32(promptTokens.count), 0, 1)
            defer { llama_batch_free(batch) }
            setBatch(&batch, tokens: promptTokens, startPosition: 0)
            #expect(llama_decode(referenceContext, batch) == 0)
            var referencePieces: [String] = []
            var position = Int32(promptTokens.count)
            for _ in 0 ..< maxTokens {
                let token = llama_sampler_sample(sampler, referenceContext, batch.n_tokens - 1)
                if llama_vocab_is_eog(vocab, token) { break }
                if let piece = piece(token, vocab: vocab) {
                    referencePieces.append(piece)
                }
                setBatch(&batch, tokens: [token], startPosition: position)
                position += 1
                guard llama_decode(referenceContext, batch) == 0 else { break }
            }

            #expect(!referencePieces.isEmpty)
            #expect(packagePieces == referencePieces)
        }

        private func makeContext(
            _ llamaModel: OpaquePointer,
            options: LlamaLanguageModel.ResolvedGenerationOptions
        ) -> OpaquePointer? {
            var params = llama_context_default_params()
            params.n_ctx = options.contextSize
            params.n_batch = options.batchSize
            params.n_threads = options.threads
            params.n_threads_batch = options.threads
            return llama_init_from_model(llamaModel, params)
        }

        private func tokenize(_ text: String, vocab: OpaquePointer) -> [llama_token]? {
            let capacity = Int32(text.utf8.count + 8)
            var tokens = [llama_token](repeating: 0, count: Int(capacity))
            let count = llama_tokenize(vocab, text, Int32(text.utf8.count), &tokens, capacity, true, true)
            guard count > 0 else { return nil }
            return Array(tokens.prefix(Int(count)))
        }

        private func piece(_ token: llama_token, vocab: OpaquePointer) -> String? {
            var buffer = [CChar](repeating: 0, count: 256)
            let count = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
            guard count > 0 else { return nil }
            return String(decoding: buffer.prefix(Int(count)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }

        private func setBatch(_ batch: inout llama_batch, tokens: [llama_token], startPosition: Int32) {
            batch.n_tokens = Int32(tokens.count)
            for (index, token) in tokens.enumerated() {
                batch.token[index] = token
                batch.pos[index] = startPosition + Int32(index)
                batch.n_seq_id[index] = 1
                if let seqIDs = batch.seq_id, let seqID = seqIDs[index] {
                    seqID[0] = 0
                }
                batch.logits[index] = index == tokens.count - 1 ? 1 : 0
            }
        }

        @Test func multimodal_rejectsImageURL() async throws {
            let session = LanguageModelSession(model: model)
            let imageSegment = Transcript.ImageSegment(url: testImageURL)
            do {
                _ = try await session.respond(to: "Describe this image", image: imageSegment)
                Issue.record("Expected error when image segments are present")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .unsupportedFeature)
            }
        }

        @Test func multimodal_rejectsImageData() async throws {
            let session = LanguageModelSession(model: model)
            let imageSegment = Transcript.ImageSegment(data: testImageData, mimeType: "image/png")
            do {
                _ = try await session.respond(to: "Describe this image", image: imageSegment)
                Issue.record("Expected error when image segments are present")
            } catch let error as LlamaLanguageModelError {
                #expect(error == .unsupportedFeature)
            }
        }

        @Test func promptExceedingBatchSize_rejected() async throws {
            let session = LanguageModelSession(model: model)

            // Use a very small batch size to test the validation
            var options = GenerationOptions(maximumResponseTokens: 10)
            options[custom: LlamaLanguageModel.self] = .init(batchSize: 8)

            // Create a prompt that will tokenize to more than 8 tokens
            // Most models will tokenize "Hello world how are you today" to more than 8 tokens
            let longPrompt = String(repeating: "Hello world how are you today? ", count: 10)

            do {
                _ = try await session.respond(to: longPrompt, options: options)
                // If we get here, either the prompt tokenized to <= 8 tokens (unlikely)
                // or the validation didn't work (bug)
                // In practice, this should throw insufficientMemory
            } catch let error as LlamaLanguageModelError {
                // Expected: prompt token count exceeds batch size
                #expect(error == .insufficientMemory)
            }
        }

        @Test(arguments: [false, true])
        func dynamicSchemaGeneration(_ streaming: Bool) async throws {
            let session = LanguageModelSession(model: model)
            let schema = try SchemaResponseTests.schema()
            let prompt = "What is the capital of France? Put the city name in answer. /no_think"
            let options = GenerationOptions(maximumResponseTokens: 128)
            let response =
                try await streaming
                ? session.streamResponse(to: prompt, schema: schema, options: options).collect()
                : session.respond(to: prompt, schema: schema, options: options)
            let answer = try SchemaResponseTests.Answer(response.content)
            #expect(!answer.answer.isEmpty)
            #expect(response.usage.output.totalTokenCount > 0)
        }

        @Test(arguments: [false, true])
        func toolRoundsKeepOneEntry(_ streaming: Bool) async throws {
            try await expectOneEntryForEachToolRound(
                model: model,
                streaming: streaming,
                prompt: "Open the front door. /no_think",
                options: GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)
            )
        }

        @Test(arguments: [false, true])
        func stoppedToolRoundKeepsItsEntry(_ streaming: Bool) async throws {
            try await expectStoppedToolRoundKeepsItsEntry(
                model: model,
                streaming: streaming,
                prompt: "Open the front door. /no_think",
                options: GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)
            )
        }

        @Test(arguments: [false, true])
        func structuredResponseWithTools(_ streaming: Bool) async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(
                model: model,
                tools: [weatherTool],
                instructions: "You are a helpful assistant. Use available tools when needed."
            )
            let prompt = "How's the weather in San Francisco? /no_think"
            let options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)

            let response =
                try await streaming
                ? session.streamResponse(to: prompt, generating: WeatherReport.self, options: options).collect()
                : session.respond(to: prompt, generating: WeatherReport.self, options: options)
            attachResponse(response)

            let calls = await weatherTool.calls
            #expect(calls.count >= 1)
            #expect(response.transcriptEntries.contains { if case .toolOutput = $0 { true } else { false } })
            // The temperature comes only from the tool's output.
            #expect(response.content.temperatureFahrenheit == 72)
        }

        @Test func structuredResponseWithToolsUsesEarlierToolOutput() async throws {
            let weatherTool = spy(on: WeatherTool())
            let call = Transcript.ToolCall(
                id: "call-1",
                toolName: weatherTool.name,
                arguments: try GeneratedContent(json: #"{"city":"Paris"}"#)
            )
            // Only the earlier tool output has the temperature;
            // the earlier answer doesn't repeat it.
            let transcript = Transcript(entries: [
                .instructions(
                    .init(
                        segments: [.text(.init(content: "You are a helpful assistant."))],
                        toolDefinitions: []
                    )
                ),
                .prompt(.init(segments: [.text(.init(content: "Look up the weather in Paris."))])),
                .toolCalls(.init([call])),
                .toolOutput(
                    .init(
                        id: call.id,
                        toolName: call.toolName,
                        segments: [.text(.init(content: "The weather in Paris is cloudy and 58°F / 14°C"))]
                    )
                ),
                .response(.init(assetIDs: [], segments: [.text(.init(content: "I looked up the weather in Paris."))])),
            ])
            let session = LanguageModelSession(model: model, tools: [weatherTool], transcript: transcript)

            let response = try await session.respond(
                to: "Report the weather in Paris that you already looked up. Don't call a tool. /no_think",
                generating: WeatherReport.self,
                options: GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)
            )
            attachResponse(response)

            #expect(await weatherTool.calls.isEmpty)
            #expect(response.content.temperatureFahrenheit == 58)
        }

        @Test(arguments: [false, true])
        func structuredResponseStoppedByDelegate(_ streaming: Bool) async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(
                model: model,
                tools: [weatherTool],
                instructions: "You are a helpful assistant. Use available tools when needed."
            )
            session.toolExecutionDelegate = StopToolExecutionDelegate()
            let prompt = "How's the weather in San Francisco? /no_think"
            let options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)

            do {
                _ =
                    try await streaming
                    ? session.streamResponse(to: prompt, generating: WeatherReport.self, options: options).collect()
                    : session.respond(to: prompt, generating: WeatherReport.self, options: options)
                Issue.record("Expected LanguageModelSession.ToolExecutionStopped")
            } catch let error as LanguageModelSession.ToolExecutionStopped {
                #expect(error.provider == "Llama")
                let calls = error.transcriptEntries.flatMap { entry -> [Transcript.ToolCall] in
                    if case .toolCalls(let calls) = entry { Array(calls) } else { [] }
                }
                #expect(calls.contains { $0.toolName == weatherTool.name })
                #expect(!error.transcriptEntries.contains { if case .toolOutput = $0 { true } else { false } })
            }
            #expect(await weatherTool.calls.isEmpty)
            // The session keeps only the prompt.
            #expect(session.transcript.count == 2)
        }

        /// An array can be empty, so a stopped response of an array type
        /// ends with an empty array and the tool calls that the delegate stopped.
        @Test(arguments: [false, true])
        func arrayResponseStoppedByDelegate(_ streaming: Bool) async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(
                model: model,
                tools: [weatherTool],
                instructions: "You are a helpful assistant. Use available tools when needed."
            )
            session.toolExecutionDelegate = StopToolExecutionDelegate()
            let prompt = "How's the weather in San Francisco? /no_think"
            let options = GenerationOptions(sampling: .greedy, maximumResponseTokens: 512)

            let response =
                try await streaming
                ? session.streamResponse(to: prompt, generating: [WeatherReport].self, options: options).collect()
                : session.respond(to: prompt, generating: [WeatherReport].self, options: options)
            #expect(response.content.isEmpty)
            let calls = response.transcriptEntries.flatMap { entry -> [Transcript.ToolCall] in
                if case .toolCalls(let calls) = entry { Array(calls) } else { [] }
            }
            #expect(calls.contains { $0.toolName == weatherTool.name })
            #expect(!response.transcriptEntries.contains { if case .toolOutput = $0 { true } else { false } })
            #expect(await weatherTool.calls.isEmpty)
        }

        @Test func structuredGenerationBasicStruct() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a person with name Alice, age 30, active status true, and score 95.5",
                generating: BasicStruct.self
            )
            attachResponse(response)
            #expect(!response.content.name.isEmpty)
            #expect(response.content.age >= 0)
        }

        @Test func structuredGenerationNestedStruct() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a person named John, age 25, living at 123 Main St, Springfield, 12345",
                generating: StructuredPerson.self
            )
            attachResponse(response)
            #expect(!response.content.name.isEmpty)
            #expect(response.content.age >= 0)
            #expect(!response.content.address.street.isEmpty)
            #expect(!response.content.address.city.isEmpty)
        }

        @Test func structuredGenerationStructWithEnum() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a task titled 'Complete project' with high priority, not completed",
                generating: TaskItem.self
            )
            attachResponse(response)
            #expect(!response.content.title.isEmpty)
            #expect([Priority.low, Priority.medium, Priority.high].contains(response.content.priority))
        }

        @Test func structuredGenerationSimpleArray() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: "Generate a list of 3 color names: red, green, blue",
                generating: SimpleArray.self
            )
            attachResponse(response)
            #expect(!response.content.colors.isEmpty)
        }

        @Test func structuredGenerationStructWithArray() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant that generates structured data."
            )
            let response = try await session.respond(
                to: """
                    Generate a quiz question:
                    - Question: What is the capital of France?
                    - Choices: London, Paris, Berlin, Madrid
                    - Answer: Paris
                    - Explanation: Paris is the capital city of France
                    """,
                generating: MultiChoiceQuestion.self
            )
            attachResponse(response)
            #expect(!response.content.text.isEmpty)
            #expect(response.content.choices.count == 4)
            #expect(!response.content.answer.isEmpty)
        }
    }
#endif  // Llama

#if Llama
    @Suite(
        "LlamaLanguageModel vision",
        .serialized,
        .enabled(
            if: ProcessInfo.processInfo.environment["LLAMA_VISION_MODEL_PATH"] != nil
                && ProcessInfo.processInfo.environment["LLAMA_VISION_MMPROJ_PATH"] != nil
        )
    )
    struct LlamaLanguageModelVisionTests {
        static let redSquarePNG = Data(
            base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAABC0lEQVR4nO3OMQ0AIAAEsfdvGhyw9gaS"
                + "CujO9j34QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E"
                + "+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E"
                + "+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E"
                + "+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E+UGcH8T5QZwfxPlBnB/E"
                + "+UHcBWwZ3g5gacwjAAAAAElFTkSuQmCC"
        )!

        let model = LlamaLanguageModel(
            modelPath: ProcessInfo.processInfo.environment["LLAMA_VISION_MODEL_PATH"]!,
            mmprojPath: ProcessInfo.processInfo.environment["LLAMA_VISION_MMPROJ_PATH"]!
        )

        @Test func describesImageData() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "What is the dominant color of this image? Answer with one word.")),
                        .image(.init(data: Self.redSquarePNG, mimeType: "image/png")),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, transcript: transcript)
            let response = try await session.respond(to: "")
            attachResponse(response)
            #expect(response.content.lowercased().contains("red"))
        }

        @Test func streamsImageDescription() async throws {
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "What is the dominant color of this image? Answer with one word.")),
                        .image(.init(data: Self.redSquarePNG, mimeType: "image/png")),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, transcript: transcript)
            let stream = session.streamResponse(to: "")
            var last = ""
            for try await snapshot in stream {
                last = snapshot.content
            }
            #expect(last.lowercased().contains("red"))
        }

        @Test func structuredResponseRejectsImagesBeforeCallingTools() async throws {
            let weatherTool = spy(on: WeatherTool())
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .text(.init(content: "Use getWeather to get the weather in the city in this image.")),
                        .image(.init(data: Self.redSquarePNG, mimeType: "image/png")),
                    ])
                )
            ])
            let session = LanguageModelSession(model: model, tools: [weatherTool], transcript: transcript)
            await #expect(throws: LlamaLanguageModelError.unsupportedFeature) {
                _ = try await session.respond(to: "", generating: WeatherReport.self)
            }
            #expect(await weatherTool.calls.isEmpty)
        }

        @Test func rejectsImagesWithoutProjector() async throws {
            let textOnlyModel = LlamaLanguageModel(
                modelPath: ProcessInfo.processInfo.environment["LLAMA_VISION_MODEL_PATH"]!
            )
            let transcript = Transcript(entries: [
                .prompt(
                    Transcript.Prompt(segments: [
                        .image(.init(data: Self.redSquarePNG, mimeType: "image/png"))
                    ])
                )
            ])
            let session = LanguageModelSession(model: textOnlyModel, transcript: transcript)
            await #expect(throws: LlamaLanguageModelError.unsupportedFeature) {
                _ = try await session.respond(to: "")
            }
        }
    }
#endif
