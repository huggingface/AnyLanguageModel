import Foundation
import Testing

@testable import AnyLanguageModel

@Generable(description: "A contact")
private struct ImplicitNilContact {
    var name: String
    var nickname: String?
    var tags: [String]?
}

@Generable(description: "A contact", representNilExplicitlyInGeneratedContent: true)
private struct ExplicitNilContact {
    var name: String
    var nickname: String?
    var tags: [String]?
}

@Generable(description: "A place", representNilExplicitlyInGeneratedContent: true)
private struct ExplicitNilPlace {
    var name: String
    var zone: String?
    var alias: String?
}

@Generable(description: "A greeting", representNilExplicitlyInGeneratedContent: true)
private struct ExplicitNilGreeting {
    var salutation: String?
    var name: String
}

@Suite("Explicit nil")
struct ExplicitNilTests {
    @Test func nilOptionalPropertiesAreLeftOutByDefault() throws {
        let contact = ImplicitNilContact(name: "Alice", nickname: nil, tags: nil)
        guard case .structure(let properties, let orderedKeys) = contact.generatedContent.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties.keys.sorted() == ["name"])
        #expect(orderedKeys == ["name"])
        #expect(contact.generatedContent == (try ImplicitNilContact(contact.generatedContent)).generatedContent)
    }

    @Test func nilOptionalPropertiesAreNullWhenExplicit() throws {
        let contact = ExplicitNilContact(name: "Alice", nickname: nil, tags: nil)
        guard case .structure(let properties, let orderedKeys) = contact.generatedContent.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
        #expect(properties["tags"]?.kind == .null)
        #expect(orderedKeys == ["name", "nickname", "tags"])
    }

    @Test func setOptionalPropertiesAreIncludedEitherWay() {
        let implicit = ImplicitNilContact(name: "Alice", nickname: "Al", tags: ["friend"])
        let explicit = ExplicitNilContact(name: "Alice", nickname: "Al", tags: ["friend"])
        #expect(implicit.generatedContent == explicit.generatedContent)
    }

    @Test func bothFormsDecode() throws {
        let omitted = try GeneratedContent(json: #"{"name": "Alice"}"#)
        let null = try GeneratedContent(json: #"{"name": "Alice", "nickname": null, "tags": null}"#)
        for content in [omitted, null] {
            let implicit = try ImplicitNilContact(content)
            let explicit = try ExplicitNilContact(content)
            #expect(implicit.nickname == nil && implicit.tags == nil)
            #expect(explicit.nickname == nil && explicit.tags == nil)
        }
    }

    @Test func flagDoesNotChangeEncodedSchema() throws {
        let properties = [
            GenerationSchema.Property(name: "name", type: String.self),
            GenerationSchema.Property(name: "nickname", type: String?.self),
        ]
        let implicit = GenerationSchema(type: ImplicitNilContact.self, properties: properties)
        let explicit = GenerationSchema(
            type: ImplicitNilContact.self,
            representNilExplicitlyInGeneratedContent: true,
            properties: properties
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        #expect(try encoder.encode(implicit) == encoder.encode(explicit))
    }

    @Test func schemaFillsInNullForOmittedOptionalProperties() throws {
        let content = try GeneratedContent(json: #"{"name": "Alice"}"#)

        let explicit = ExplicitNilContact.generationSchema.representingNilExplicitly(in: content)
        guard case .structure(let properties, _) = explicit.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
        #expect(properties["tags"]?.kind == .null)
        #expect(properties["name"] == GeneratedContent("Alice"))

        #expect(ImplicitNilContact.generationSchema.representingNilExplicitly(in: content) == content)
    }

    @Test func dynamicSchemaFillsInNullForOmittedOptionalProperties() throws {
        let contact = DynamicGenerationSchema(
            name: "Contact",
            representNilExplicitlyInGeneratedContent: true,
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "nickname", schema: .init(type: String.self), isOptional: true),
            ]
        )
        let schema = try GenerationSchema(root: contact, dependencies: [])
        let content = schema.representingNilExplicitly(in: try GeneratedContent(json: #"{"name": "Alice"}"#))
        guard case .structure(let properties, _) = content.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
    }

    @Test func nullableUnionFillsInNullForTheObjectVariant() throws {
        let contact = DynamicGenerationSchema(
            name: "Contact",
            representNilExplicitlyInGeneratedContent: true,
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "nickname", schema: .init(type: String.self), isOptional: true),
            ]
        )
        let nullableContact = DynamicGenerationSchema(name: "NullableContact", anyOf: [contact, .null])
        let schema = try GenerationSchema(root: nullableContact, dependencies: [])

        let content = schema.representingNilExplicitly(in: try GeneratedContent(json: #"{"name": "Alice"}"#))
        guard case .structure(let properties, _) = content.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)

        let null = GeneratedContent(kind: .null)
        #expect(schema.representingNilExplicitly(in: null) == null)
    }

    @Test func unionSkipsObjectVariantsWithMissingRequiredProperties() throws {
        let person = DynamicGenerationSchema(
            name: "Person",
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "age", schema: .init(type: Int.self)),
            ]
        )
        let contact = DynamicGenerationSchema(
            name: "Contact",
            representNilExplicitlyInGeneratedContent: true,
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "nickname", schema: .init(type: String.self), isOptional: true),
            ]
        )
        let either = DynamicGenerationSchema(name: "Either", anyOf: [person, contact])
        let schema = try GenerationSchema(root: either, dependencies: [])

        let content = schema.representingNilExplicitly(in: try GeneratedContent(json: #"{"name": "Alice"}"#))
        guard case .structure(let properties, _) = content.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
    }

    @Test func nestedUnionFillsInNullForTheObjectVariant() throws {
        let contact = DynamicGenerationSchema(
            name: "Contact",
            representNilExplicitlyInGeneratedContent: true,
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "nickname", schema: .init(type: String.self), isOptional: true),
            ]
        )
        let nullableContact = DynamicGenerationSchema(name: "NullableContact", anyOf: [contact, .null])
        let outer = DynamicGenerationSchema(
            name: "Outer",
            anyOf: [
                DynamicGenerationSchema(type: String.self),
                DynamicGenerationSchema(referenceTo: "NullableContact"),
            ]
        )
        let schema = try GenerationSchema(root: outer, dependencies: [nullableContact])

        let content = schema.representingNilExplicitly(in: try GeneratedContent(json: #"{"name": "Alice"}"#))
        guard case .structure(let properties, _) = content.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
    }

    @Test func filledInPropertiesFollowDeclarationOrder() throws {
        let content = try GeneratedContent(json: #"{"name": "Home"}"#)
        let filled = ExplicitNilPlace.generationSchema.representingNilExplicitly(in: content)
        #expect(filled == ExplicitNilPlace(name: "Home", zone: nil, alias: nil).generatedContent)
    }

    @Test func filledInPropertiesKeepDeclarationOrderAroundPresentProperties() throws {
        let content = try GeneratedContent(json: #"{"name": "Alice"}"#)
        let filled = ExplicitNilGreeting.generationSchema.representingNilExplicitly(in: content)
        #expect(filled == ExplicitNilGreeting(salutation: nil, name: "Alice").generatedContent)
    }

    @Test func propertyOrderIsPartOfExplicitNilSchemaEquality() {
        let name = GenerationSchema.Property(name: "name", type: String.self)
        let nickname = GenerationSchema.Property(name: "nickname", type: String?.self)
        let first = GenerationSchema(
            type: ImplicitNilContact.self,
            representNilExplicitlyInGeneratedContent: true,
            properties: [name, nickname]
        )
        let second = GenerationSchema(
            type: ImplicitNilContact.self,
            representNilExplicitlyInGeneratedContent: true,
            properties: [nickname, name]
        )
        #expect(first != second)
        #expect(
            GenerationSchema(type: ImplicitNilContact.self, properties: [name, nickname])
                == GenerationSchema(type: ImplicitNilContact.self, properties: [nickname, name])
        )
    }

    @Test func flagIsPartOfSchemaEquality() {
        let properties = [GenerationSchema.Property(name: "nickname", type: String?.self)]
        let implicit = GenerationSchema(type: ImplicitNilContact.self, properties: properties)
        let explicit = GenerationSchema(
            type: ImplicitNilContact.self,
            representNilExplicitlyInGeneratedContent: true,
            properties: properties
        )
        #expect(implicit != explicit)
    }

    @Test func streamedResponseCollectsNullForOmittedOptionalProperties() async throws {
        let model = MockLanguageModel { _, _ in #"{"name": "Alice"}"# }
        let session = LanguageModelSession(model: model)

        let response = try await session.streamResponse(to: "Who?", generating: ExplicitNilContact.self).collect()
        guard case .structure(let properties, _) = response.rawContent.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
        #expect(lastResponseText(in: session)?.contains(#""nickname":null"#) == true)
    }

    @Test func schemaResponsesReturnNullInContent() async throws {
        let contact = DynamicGenerationSchema(
            name: "Contact",
            representNilExplicitlyInGeneratedContent: true,
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "nickname", schema: .init(type: String.self), isOptional: true),
            ]
        )
        let schema = try GenerationSchema(root: contact, dependencies: [])
        let model = MockLanguageModel { _, _ in #"{"name": "Alice"}"# }

        let response = try await LanguageModelSession(model: model).respond(to: "Who?", schema: schema)
        #expect(response.content == response.rawContent)
        guard case .structure(let properties, _) = response.content.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)

        let streamed = try await LanguageModelSession(model: model).streamResponse(to: "Who?", schema: schema)
            .collect()
        #expect(streamed.content == streamed.rawContent)
    }

    @Test func finalStreamedSnapshotHasNullInPartialContent() async throws {
        // Providers build partial content from the raw JSON, without normalizing it.
        let session = LanguageModelSession(model: RawPartialStreamingModel(json: #"{"name": "Alice"}"#))

        var last: LanguageModelSession.ResponseStream<ExplicitNilContact>.Snapshot?
        for try await snapshot in session.streamResponse(to: "Who?", generating: ExplicitNilContact.self) {
            last = snapshot
        }
        let snapshot = try #require(last)
        guard case .structure(let properties, _) = snapshot.content.generatedContent.kind else {
            Issue.record("Expected structured content")
            return
        }
        #expect(properties["nickname"]?.kind == .null)
        #expect(snapshot.content.generatedContent == snapshot.rawContent)
    }

    @Test func sessionRecordsNullForOmittedOptionalProperties() async throws {
        let model = MockLanguageModel { _, _ in #"{"name": "Alice"}"# }

        let explicitSession = LanguageModelSession(model: model)
        let explicit = try await explicitSession.respond(to: "Who?", generating: ExplicitNilContact.self)
        #expect(explicit.rawContent.jsonString.contains("nickname"))
        #expect(lastResponseText(in: explicitSession)?.contains(#""nickname":null"#) == true)

        let implicitSession = LanguageModelSession(model: model)
        let implicit = try await implicitSession.respond(to: "Who?", generating: ImplicitNilContact.self)
        #expect(!implicit.rawContent.jsonString.contains("nickname"))
        #expect(lastResponseText(in: implicitSession)?.contains("nickname") == false)
    }

    private func lastResponseText(in session: LanguageModelSession) -> String? {
        guard case .response(let response)? = session.transcript.last,
            case .text(let text)? = response.segments.first
        else { return nil }
        return text.content
    }
}

/// A model that streams one snapshot whose partial content comes straight from the raw JSON.
private struct RawPartialStreamingModel: LanguageModel {
    typealias UnavailableReason = Never

    let json: String

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        fatalError("Not used")
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let json = json
        let stream = AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
            continuation in
            do {
                let rawContent = try GeneratedContent(json: json)
                continuation.yield(
                    .init(content: try Content.PartiallyGenerated(rawContent), rawContent: rawContent)
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return LanguageModelSession.ResponseStream(stream: stream)
    }
}
