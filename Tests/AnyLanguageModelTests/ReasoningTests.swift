import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Display reasoning")
struct ReasoningTests {
    @Test func existingInitializersDefaultToNil() async throws {
        let raw = GeneratedContent("Answer")
        let response = LanguageModelSession.Response(content: "Answer", rawContent: raw, transcriptEntries: [])
        let snapshot = LanguageModelSession.ResponseStream<String>.Snapshot(content: "Answer", rawContent: raw)
        let stream = LanguageModelSession.ResponseStream(content: "Answer", rawContent: raw)
        #expect(response.reasoning == nil)
        #expect(snapshot.reasoning == nil)
        #expect(try await stream.collect().reasoning == nil)
    }

    @Test func singleValueAndStructuredWrapperPreserveReasoning() async throws {
        let stream = LanguageModelSession.ResponseStream(
            content: "Answer",
            rawContent: GeneratedContent("Answer"),
            reasoning: "Display summary"
        )
        #expect(try await stream.collect().reasoning == "Display summary")
        let wrapped: LanguageModelSession.ResponseStream<String> = ReasoningModel().streamStructuredResponse {
            .init(
                content: "Answer",
                rawContent: GeneratedContent("Answer"),
                transcriptEntries: [],
                reasoning: "Display summary"
            )
        }
        #expect(try await wrapped.collect().reasoning == "Display summary")
    }

    @Test func sessionYieldsReasoningOnlyChangesAndCollectPreservesFinalValue() async throws {
        let session = LanguageModelSession(model: ReasoningModel())
        var reasoning: [String?] = []
        var content: [String] = []
        for try await snapshot in session.streamResponse(to: "Question") {
            reasoning.append(snapshot.reasoning)
            content.append(snapshot.content)
        }
        #expect(reasoning == ["First", "First second", "First second"])
        #expect(content == ["", "", "Answer"])
        try assertAnswerOnly(in: session)
        let response = try await session.streamResponse(to: "Again").collect()
        #expect(response.content == "Answer")
        #expect(response.reasoning == "First second")
        try assertAnswerOnly(in: session)
    }

    @Test func respondPreservesReasoningWithoutPersistingIt() async throws {
        let session = LanguageModelSession(model: ReasoningModel())
        let response = try await session.respond(to: "Question")
        #expect(response.content == "Answer")
        #expect(response.reasoning == "First second")
        try assertAnswerOnly(in: session)
    }

    @Test func schemaOverloadsPreserveReasoning() async throws {
        let session = LanguageModelSession(model: ReasoningModel())
        let response = try await session.respond(to: "Question", schema: String.generationSchema)
        #expect(response.reasoning == "First second")
        #expect(response.rawContent == GeneratedContent("Answer"))
        let streamed = try await session.streamResponse(to: "Again", schema: String.generationSchema).collect()
        #expect(streamed.reasoning == "First second")
        #expect(streamed.rawContent == GeneratedContent("Answer"))
        try assertAnswerOnly(in: session)
    }

    @Test func cancelledReasoningStreamDoesNotCommitAssistantResponse() async throws {
        let (stopped, stopSignal) = AsyncStream<Void>.makeStream()
        let session = LanguageModelSession(model: ReasoningModel(stopSignal: stopSignal))
        let consumer = Task {
            for try await snapshot in session.streamResponse(to: "Question") {
                #expect(snapshot.content == "")
                #expect(snapshot.reasoning == "First")
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        _ = await consumer.result
        for await _ in stopped {}
        // The provider termination callback runs before the relay finishes cleanup.
        while session.isResponding { await Task.yield() }
        #expect(session.transcript.count == 1)
        #expect(!session.isResponding)
    }

    private func assertAnswerOnly(in session: LanguageModelSession) throws {
        let responses = session.transcript.compactMap { entry -> Transcript.Response? in
            if case .response(let response) = entry { return response }
            return nil
        }
        #expect(!responses.isEmpty)
        for response in responses {
            #expect(response.segments.count == 1)
            guard case .text(let text) = response.segments.first else {
                Issue.record("Expected only an answer text segment")
                return
            }
            #expect(text.content == "Answer")
        }
        let encoded = try JSONEncoder().encode(session.transcript)
        #expect(!String(decoding: encoded, as: UTF8.self).contains("First"))
    }
}

private struct ReasoningModel: LanguageModel {
    typealias UnavailableReason = Never
    var stopSignal: AsyncStream<Void>.Continuation? = nil

    func respond<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        try await streamResponse(
            within: session,
            to: prompt,
            generating: type,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        ).collect()
    }

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        .init(
            stream: AsyncThrowingStream { continuation in
                do {
                    if let stopSignal {
                        continuation.onTermination = { _ in stopSignal.finish() }
                        let raw = GeneratedContent("")
                        continuation.yield(
                            .init(
                                content: try Content(raw).asPartiallyGenerated(),
                                rawContent: raw,
                                reasoning: "First"
                            )
                        )
                        return
                    }
                    for (text, reasoning) in [("", "First"), ("", "First second"), ("Answer", "First second")] {
                        let raw = GeneratedContent(text)
                        continuation.yield(
                            .init(
                                content: try Content(raw).asPartiallyGenerated(),
                                rawContent: raw,
                                reasoning: reasoning
                            )
                        )
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        )
    }
}
