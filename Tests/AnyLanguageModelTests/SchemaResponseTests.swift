import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Schema responses")
struct SchemaResponseTests {
    @Generable struct Answer {
        var answer: String
    }

    static func schema() throws -> GenerationSchema {
        try GenerationSchema(
            root: DynamicGenerationSchema(
                name: "Answer",
                properties: [.init(name: "answer", schema: .init(type: String.self))]
            ),
            dependencies: []
        )
    }

    private func encoded<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return try encoder.encode(value)
    }

    private func checkTranscript(_ session: LanguageModelSession, schema: GenerationSchema) throws {
        let prompt = try #require(
            session.transcript.compactMap { entry -> Transcript.Prompt? in
                if case .prompt(let prompt) = entry { return prompt }
                return nil
            }.last
        )
        let format = try #require(prompt.responseFormat)
        #expect(try encoded(format) == encoded(Transcript.ResponseFormat(schema: schema)))
        #expect(session.transcript.contains { if case .response = $0 { true } else { false } })
        #expect(!session.isResponding)
    }

    @Test(arguments: 0 ..< 6, [false, true])
    func schemaOverloads(_ overload: Int, _ includeSchema: Bool) async throws {
        let schema = try Self.schema()
        var model = MockLanguageModel.fixed(#"{"answer":"Paris"}"#)
        model.usage = .init(
            input: .init(totalTokenCount: 3, cachedTokenCount: 0),
            output: .init(totalTokenCount: 2, reasoningTokenCount: 0)
        )
        let session = LanguageModelSession(model: model)
        let options = GenerationOptions(temperature: 0.25, maximumResponseTokens: 42)
        let response: LanguageModelSession.Response<GeneratedContent>
        switch overload {
        case 0:
            response = try await session.respond(
                to: Prompt("Capital of France?"),
                schema: schema,
                includeSchemaInPrompt: includeSchema,
                options: options
            )
        case 1:
            response = try await session.respond(
                to: "Capital of France?",
                schema: schema,
                includeSchemaInPrompt: includeSchema,
                options: options
            )
        case 2:
            response = try await session.respond(schema: schema, includeSchemaInPrompt: includeSchema, options: options)
            {
                "Capital of France?"
            }
        case 3:
            response = try await session.streamResponse(
                to: Prompt("Capital of France?"),
                schema: schema,
                includeSchemaInPrompt: includeSchema,
                options: options
            ).collect()
        case 4:
            response = try await session.streamResponse(
                to: "Capital of France?",
                schema: schema,
                includeSchemaInPrompt: includeSchema,
                options: options
            ).collect()
        default:
            response = try await session.streamResponse(
                schema: schema,
                includeSchemaInPrompt: includeSchema,
                options: options
            ) {
                "Capital of France?"
            }.collect()
        }
        #expect(try response.content.value(String.self, forProperty: "answer") == "Paris")
        let requests = model.requests.withLock { $0 }
        #expect(requests.count == 1)
        let request = try #require(requests.first)
        #expect(try encoded(request.schema) == encoded(schema))
        #expect(request.includeSchemaInPrompt == includeSchema)
        #expect(request.options.temperature == 0.25)
        #expect(request.options.maximumResponseTokens == 42)
        #expect(session.usage == model.usage)
        try checkTranscript(session, schema: schema)
    }

    @Test(arguments: [false, true])
    func typedGenerationPreservesSchema(_ streaming: Bool) async throws {
        let model = MockLanguageModel.fixed(#"{"answer":"Paris"}"#)
        let session = LanguageModelSession(model: model)
        let response =
            try await streaming
            ? session.streamResponse(to: "Capital of France?", generating: Answer.self).collect()
            : session.respond(to: "Capital of France?", generating: Answer.self)
        #expect(response.content.answer == "Paris")
        let request = try #require(model.requests.withLock { $0.first })
        #expect(try encoded(request.schema) == encoded(Answer.generationSchema))
        try checkTranscript(session, schema: Answer.generationSchema)
    }

    @Test(arguments: [false, true])
    func freeTextHasNoResponseFormat(_ streaming: Bool) async throws {
        let session = LanguageModelSession(model: MockLanguageModel.fixed("Paris"))
        _ =
            try await streaming
            ? session.streamResponse(to: "Capital of France?").collect()
            : session.respond(to: "Capital of France?")
        for case .prompt(let prompt) in session.transcript {
            #expect(prompt.responseFormat == nil)
        }
    }

    // This conformer intentionally implements only the original requirements.
    private struct LegacyModel: LanguageModel {
        typealias UnavailableReason = Never
        let mock: MockLanguageModel

        func respond<Content: Generable>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> {
            try await mock.respond(
                within: session,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        func streamResponse<Content: Generable>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> {
            mock.streamResponse(
                within: session,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }
    }

    @Test(arguments: [false, true])
    func legacyConformerUsesOneWayDefaults(_ streaming: Bool) async throws {
        let mock = MockLanguageModel.fixed(#"{"answer":"Paris"}"#)
        let session = LanguageModelSession(model: LegacyModel(mock: mock))
        let schema = try Self.schema()
        let response =
            try await streaming
            ? session.streamResponse(to: "Capital of France?", schema: schema).collect()
            : session.respond(to: "Capital of France?", schema: schema)
        #expect(try response.content.value(String.self, forProperty: "answer") == "Paris")
        let request = try #require(mock.requests.withLock { $0.first })
        #expect(try encoded(request.schema) == encoded(GeneratedContent.generationSchema))
    }
}
