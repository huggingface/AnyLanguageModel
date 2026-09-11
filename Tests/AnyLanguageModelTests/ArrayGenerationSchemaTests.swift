import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Array Generation Schema")
struct ArrayGenerationSchemaTests {
    @Generable
    struct Item {
        var name: String
        var count: Int
    }

    @Generable
    struct Response {
        var title: String
        var items: [Item]
    }

    @Generable
    struct NestedItem {
        var item: Item
        var relatedItems: [Item]
    }

    @Generable
    struct OptionalResponse {
        @Guide(description: "Suggested items", .count(2))
        var items: [NestedItem]?
    }

    @Test func arraysPreserveElementDefinitions() throws {
        try checkArraySchema(Item.self)
        try checkArraySchema(NestedItem.self)
        try checkArraySchema([Item].self)
        try checkArraySchema(String.self)
    }

    private func checkArraySchema<Element: Generable>(_ type: Element.Type) throws {
        let elementSchema = Element.generationSchema
        let schema = [Element].generationSchema
        guard case .array(let array) = schema.root else {
            Issue.record("Expected an array schema for \(Element.self)")
            return
        }
        #expect(array.items == elementSchema.root)
        #expect(schema.defs == elementSchema.defs)

        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(GenerationSchema.self, from: data)
        #expect(decoded == schema)
        #expect(decoded.defs == elementSchema.defs)
    }

    @Test func objectArrayPropertyEncodesElementDefinition() throws {
        let data = try JSONEncoder().encode(Response.generationSchema)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let definitions = try #require(json["$defs"] as? [String: Any])
        let response = try #require(definitions[String(reflecting: Response.self)] as? [String: Any])
        let properties = try #require(response["properties"] as? [String: Any])
        let array = try #require(properties["items"] as? [String: Any])
        #expect(array["type"] as? String == "array")
        let items = try #require(array["items"] as? [String: Any])
        let itemName = String(reflecting: Item.self)
        #expect(items["$ref"] as? String == "#/$defs/\(itemName)")

        let item = try #require(definitions[itemName] as? [String: Any])
        let itemProperties = try #require(item["properties"] as? [String: Any])
        #expect((itemProperties["name"] as? [String: Any])?["type"] as? String == "string")
        #expect((itemProperties["count"] as? [String: Any])?["type"] as? String == "integer")
        #expect(Set(item["required"] as? [String] ?? []) == ["name", "count"])
    }

    @Test func optionalArrayPreservesTransitiveDefinitionsAndGuides() throws {
        let schema = OptionalResponse.generationSchema
        let root = try #require(schema.withResolvedRoot())
        guard case .object(let object) = root.root,
            case .array(let array) = object.properties["items"]
        else {
            Issue.record("Expected an object with an array property")
            return
        }
        #expect(!object.required.contains("items"))
        #expect(array.description == "Suggested items")
        #expect(array.minItems == 2)
        #expect(array.maxItems == 2)
        #expect(array.items == NestedItem.generationSchema.root)
        for (name, definition) in NestedItem.generationSchema.defs {
            #expect(schema.defs[name] == definition)
        }
        let itemName = String(reflecting: Item.self)
        #expect(schema.defs[itemName] == Item.generationSchema.defs[itemName])
    }
}
