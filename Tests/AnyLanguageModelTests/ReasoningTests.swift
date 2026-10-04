import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Reasoning transcript")
struct ReasoningTests {
    @Test func codablePreservesOpaqueSignatureAndMetadata() throws {
        let reasoning = Transcript.Reasoning(
            id: "reason",
            metadata: ["provider": GeneratedContent("fixture"), "redacted": GeneratedContent(true)],
            segments: [.text(.init(id: "segment", content: "Summary"))],
            signature: Data([0, 255, 16])
        )
        let transcript = Transcript(entries: [.reasoning(reasoning)])
        #expect(try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(transcript)) == transcript)
        #expect(transcript.first?.id == "reason")
        #expect(!transcript.first!.description.contains("255"))
    }

    @Test func originalInitializerFunctionReferencesCompile() async throws {
        guard #available(macOS 26.0, iOS 26.0, watchOS 27.0, *) else { return }
        let responseInit = LanguageModelSession.Response<String>.init(content:rawContent:transcriptEntries:usage:)
        let snapshotInit = LanguageModelSession.ResponseStream<String>.Snapshot.init(
            content:
            rawContent:
            transcriptEntries:
            usage:
        )
        let streamInit = LanguageModelSession.ResponseStream<String>.init(content:rawContent:usage:)
        let raw = GeneratedContent("Answer")
        #expect(responseInit("Answer", raw, [], .zero).content == "Answer")
        #expect(snapshotInit("Answer", raw, [], .zero).content == "Answer")
        #expect(try await streamInit("Answer", raw, .zero).collect().content == "Answer")
    }

    @Test func cumulativeEntriesPersistOnceAndDoNotPolluteAnswer() async throws {
        let session = LanguageModelSession(model: ReasoningModel())
        var ids: [String] = []
        for try await snapshot in session.streamResponse(to: "Question") {
            ids.append(try #require(snapshot.transcriptEntries.first?.id))
            let visible: String = snapshot.content
            #expect(!visible.contains("First"))
        }
        #expect(Set(ids) == ["reason"])
        #expect(session.transcript.count == 3)
        guard case .reasoning(let reasoning) = session.transcript[1] else { Issue.record("Missing reasoning"); return }
        #expect(reasoning.segments == [.text(.init(id: "segment", content: "First second"))])
        guard case .response(let answer) = session.transcript[2] else { Issue.record("Missing answer"); return }
        #expect(answer.segments.count == 1)
        #expect(answer.segments.first?.description == "Answer")
    }

    @Test func nonstreamAndSchemaRetainEntries() async throws {
        let session = LanguageModelSession(model: ReasoningModel())
        let response = try await session.respond(to: "Question", schema: String.generationSchema)
        #expect(response.transcriptEntries.count == 1)
        #expect(response.rawContent == GeneratedContent("Answer"))
        let streamed = try await session.streamResponse(to: "Again", schema: String.generationSchema).collect()
        #expect(streamed.transcriptEntries.count == 1)
        #expect(streamed.rawContent == GeneratedContent("Answer"))
    }

}

private struct ReasoningModel: LanguageModel {
    typealias UnavailableReason = Never

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
                    for (text, reasoning) in [("", "First"), ("", "First second"), ("Answer", "First second")] {
                        let raw = GeneratedContent(text)
                        let entries: [Transcript.Entry] = [
                            .reasoning(
                                .init(
                                    id: "reason",
                                    segments: [.text(.init(id: "segment", content: reasoning))]
                                )
                            )
                        ]
                        continuation.yield(
                            .init(
                                content: try Content(raw).asPartiallyGenerated(),
                                rawContent: raw,
                                transcriptEntries: ArraySlice(entries)
                            )
                        )
                    }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
        )
    }
}
