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
